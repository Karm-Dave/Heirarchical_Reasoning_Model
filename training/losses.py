def act_loss(outputs, targets, halting_probs, ponder_cost):
    loss = 0.0
    expected_steps = 0.0

    for y_hat, p in zip(outputs, halting_probs):
        loss += p * nn.functional.cross_entropy(y_hat, targets)
        expected_steps += p
    
    return loss + ponder_cost * expected_steps