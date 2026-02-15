import torch

from training.losses import act_loss


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

            if cfg.halting.enabled and model.halting is not None:
                halting_probs, expected_steps = model.halting(h_states)
            else:
                halting_probs = [o.new_zeros((o.size(0), 1)) for o in outputs]
                halting_probs[-1] = halting_probs[-1] + 1.0
                expected_steps = None

            loss = act_loss(
                outputs=outputs,
                targets=y,
                halting_probs=halting_probs,
                ponder_cost=cfg.halting.ponder_cost,
                expected_steps=expected_steps,
            )

            logits = outputs[-1]
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
