"""Federated strategies reproducing FLamby's FedAvg, FedProx and SCAFFOLD (commit edacf54).

FLamby conventions kept on purpose:
  * each client keeps its own model copy; only *parameters* are averaged (weighted by the
    number of training images), so BatchNorm running statistics stay local to each client;
  * one round = `num_updates` mini-batches per client from an endlessly cycling loader;
  * the reported "global" model is client 0's copy (FLamby evaluates `s.run()[0]`).
FedBN (Li et al., 2021): BatchNorm weights and biases are not averaged either; each client keeps its
own, and each center is evaluated with its own client's model.
FedPer (Arivazhagan et al., 2019): the parameters named in `local_names` (the classifier head) are not
averaged; each client keeps its own head, the body is averaged as in FedAvg, and each center is
evaluated with its own client's model.
Mixed precision (AMP) is optional; gradients are unscaled before the FedProx / SCAFFOLD terms.
"""
import copy

import torch


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Client:
    def __init__(self, model, loss_fn, loader, lr, device, amp):
        self.model = copy.deepcopy(model).to(device)
        self.loss_fn = loss_fn.to(device)
        self.loader = loader
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.device = device
        self.amp = amp and device.type == "cuda"
        self.scaler = make_scaler(self.amp)
        self.last_loss = float("nan")

    def params(self):
        return list(self.model.parameters())

    def local_train(self, num_updates, mu=0.0, anchor=None, correction=None):
        self.model.train()
        total = 0.0
        for _ in range(num_updates):
            x, y = self.loader.next()
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                out = self.model(x)
            loss = self.loss_fn(out.float(), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            with torch.no_grad():
                if mu > 0 and anchor is not None:          # FedProx: grad of mu/2 ||w - w0||^2
                    for p, p0 in zip(self.params(), anchor):
                        if p.grad is not None:
                            p.grad.add_(p - p0, alpha=mu)
                if correction is not None:                  # SCAFFOLD (FLamby form): grad - c_i
                    for p, c in zip(self.params(), correction):
                        if p.grad is not None:
                            p.grad.sub_(c)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            total += loss.item()
        self.last_loss = total / max(num_updates, 1)


class Federation:
    def __init__(self, model, loss_fn, loaders, lr, device, amp=True, strategy="fedavg",
                 mu=0.0, server_lr=1.0, local_names=None):
        self.clients = [Client(model, loss_fn, l, lr, device, amp) for l in loaders]
        self.sizes = [l.n for l in loaders]
        self.N = float(sum(self.sizes))
        self.strategy = strategy
        self.mu = mu
        self.lr = lr
        self.server_lr = server_lr
        if strategy == "fedbn":                         # FedBN: BatchNorm weights and biases stay local
            bn_names = {f"{mn}.{pn}" for mn, m in self.clients[0].model.named_modules()
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                        for pn, _ in m.named_parameters(recurse=False)}
            self.is_local = [n in bn_names for n, _ in self.clients[0].model.named_parameters()]
            assert any(self.is_local), "no BatchNorm parameters found"
        elif strategy == "fedper":                      # FedPer: the named head parameters stay local
            names = [n for n, _ in self.clients[0].model.named_parameters()]
            assert local_names and set(local_names) <= set(names), f"unknown local parameters {local_names}"
            self.is_local = [n in set(local_names) for n in names]
        if strategy == "scaffold":
            init = [p.detach().clone() for p in self.clients[0].params()]
            self.prev_client_state = [[t.clone() for t in init] for _ in self.clients]
            self.corrections = [[torch.zeros_like(t) for t in init] for _ in self.clients]

    @torch.no_grad()
    def _snapshot(self, c):
        return [p.detach().clone() for p in c.params()]

    @torch.no_grad()
    def _restore(self, c, state):
        for p, s in zip(c.params(), state):
            p.copy_(s)

    def round(self, num_updates):
        deltas, own = [], []
        for i, c in enumerate(self.clients):
            server_state = self._snapshot(c)
            if self.strategy == "scaffold":
                denom = self.server_lr * self.lr * num_updates
                new_corr = [ci + (s - q) / denom for ci, s, q in
                             zip(self.corrections[i], server_state, self.prev_client_state[i])]
                self.corrections[i] = new_corr
                c.local_train(num_updates, correction=new_corr)
                with torch.no_grad():
                    nxt = [self.server_lr * p.detach() + (1 - self.server_lr) * s
                           for p, s in zip(c.params(), server_state)]
                self.prev_client_state[i] = nxt
            elif self.strategy == "fedprox":
                c.local_train(num_updates, mu=self.mu, anchor=server_state)
                nxt = self._snapshot(c)
            else:
                c.local_train(num_updates)
                nxt = self._snapshot(c)
            deltas.append([(n - s) * self.sizes[i] for n, s in zip(nxt, server_state)])
            own.append(nxt)
            self._restore(c, server_state)
        with torch.no_grad():
            agg = [sum(d[k] for d in deltas) / self.N for k in range(len(deltas[0]))]
            for i, c in enumerate(self.clients):
                for k, (p, a) in enumerate(zip(c.params(), agg)):
                    if self.strategy in ("fedbn", "fedper") and self.is_local[k]:
                        p.copy_(own[i][k])              # keep this client's own (BatchNorm / head) parameters
                    else:
                        p.add_(a)
        return [c.last_loss for c in self.clients]


@torch.no_grad()
def personal_check(fed):
    """Largest |difference| between client 0 and the others, for shared and for local parameters
    (FedPer / FedBN sanity check: shared must be 0, local must be > 0)."""
    names = [n for n, _ in fed.clients[0].model.named_parameters()]
    shared = local = 0.0
    for c in fed.clients[1:]:
        for k, (p0, p) in enumerate(zip(fed.clients[0].params(), c.params())):
            d = (p0 - p).abs().max().item()
            if fed.is_local[k]:
                local = max(local, d)
            else:
                shared = max(shared, d)
    return shared, local, [n for n, l in zip(names, fed.is_local) if l]


def train_pooled(model, loss_fn, loader, epochs, lr, device, amp=True, log=print):
    """FLamby's pooled baseline in the FL benchmark: Adam, fixed epochs, final model kept."""
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    amp = amp and device.type == "cuda"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = make_scaler(amp)
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(y)
            n += len(y)
        log(f"  epoch {ep + 1}/{epochs}  train loss {tot / max(n, 1):.4f}")
    return model
