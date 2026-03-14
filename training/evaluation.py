import torch

from training.losses import act_loss


def _build_halting(cfg, model, outputs, h_states):
    adaptive = bool(getattr(cfg.halting, "adaptive", True))
    if cfg.halting.enabled and adaptive and model.halting is not None:
        return model.halting(h_states)

    if cfg.halting.enabled and not adaptive:
        n = len(outputs)
        p = 1.0 / max(n, 1)
        halting_probs = [o.new_full((o.size(0), 1), p) for o in outputs]
        expected_steps = None
        return halting_probs, expected_steps

    halting_probs = [o.new_zeros((o.size(0), 1)) for o in outputs]
    halting_probs[-1] = halting_probs[-1] + 1.0
    expected_steps = None
    return halting_probs, expected_steps


def evaluate(model, dataloader, device, cfg):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs, h_states = model(x)
            halting_probs, expected_steps = _build_halting(cfg, model, outputs, h_states)
            outputs_for_loss = outputs[: len(halting_probs)]

            loss = act_loss(
                outputs=outputs_for_loss,
                targets=y,
                halting_probs=halting_probs,
                ponder_cost=cfg.halting.ponder_cost,
                expected_steps=expected_steps,
            )

            logits = outputs_for_loss[-1]
            pred = torch.argmax(logits, dim=-1)

            batch_size = y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((pred == y).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "samples": 0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }
