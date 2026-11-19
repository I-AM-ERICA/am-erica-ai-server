# 역할: 생성 action 토큰의 GAE, PPO clipping, Value clipping 수식을 구현합니다.
import torch


@torch.no_grad()
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    # 검색 관측 토큰을 제거한 action 시간축입니다. 모든 종료(길이 제한 포함)를 terminal로 정의합니다.
    if rewards.ndim != 1 or rewards.shape != values.shape or not rewards.numel():
        raise ValueError("rewards/values must be nonempty 1-D tensors with equal shape")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device)
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
    return advantages, advantages + values


def policy_loss(logp, old_logp, advantages, clip_range):
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)
    metrics = {"approx_kl": ((ratio - 1) - log_ratio).detach().mean().item(),
               "clip_fraction": ((ratio - 1).abs() > clip_range).float().mean().item()}
    return loss, metrics


def value_loss(values, old_values, returns, clip_range):
    clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
    return 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square())
