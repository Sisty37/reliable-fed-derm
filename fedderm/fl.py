"""Federated strategies reproducing FLamby's FedAvg, FedProx and SCAFFOLD (commit edacf54).

FLamby conventions kept on purpose:
  * each client keeps its own model copy; only *parameters* are averaged (weighted by the
    number of training images), so BatchNorm running statistics stay local to each client;
  * one round = `num_updates` mini-batches per client from an endlessly cycling loader;
  * the reported "global" model is client 0's copy (FLamby evaluates `s.run()[0]`).
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
                 mu=0.0, server_lr=1.0):
        self.clients = [Client(model, loss_fn, l, lr, device, amp) for l in loaders]
        self.sizes = [l.n for l in loaders]
        self.N = float(sum(self.sizes))
        self.strategy = strategy
        self.mu = mu
        self.lr = lr
        self.server_lr = server_lr
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
        deltas = []
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
            self._restore(c, server_state)
        with torch.no_grad():
            agg = [sum(d[k] for d in deltas) / self.N for k in range(len(deltas[0]))]
            for c in self.clients:
                for p, a in zip(c.params(), agg):
                    p.add_(a)
        return [c.last_loss for c in self.clients]


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
