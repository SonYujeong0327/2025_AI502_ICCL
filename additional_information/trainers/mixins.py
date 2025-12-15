from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from models.example_adapter import AdapterSchedule, ExampleAdapterForCausalLM


@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5  # weight on distillation loss
    hard_label_weight: Optional[float] = None  # overrides 1 - alpha when provided


class DistillationTrainerMixin:
    """
    Mixin that augments a Trainer with knowledge distillation loss computation.
    The consuming class must define `self.teacher_model` and `self.distillation_config`
    along with an initialized KL divergence loss as `self.kl_div`.
    """

    teacher_model: nn.Module
    distillation_config: DistillationConfig
    kl_div: nn.KLDivLoss

    def _init_teacher(self, teacher_model: nn.Module, device: torch.device) -> None:
        if teacher_model is None:
            raise ValueError("teacher_model must be provided when using DistillationTrainerMixin.")
        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        hard_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.distillation_config
        temperature = cfg.temperature

        soft_targets = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        kd_loss = self.kl_div(soft_targets, teacher_probs) * (temperature ** 2)

        if hard_loss is None:
            if labels is not None:
                hard_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                hard_loss = torch.tensor(0.0, device=student_logits.device)

        hard_weight = cfg.hard_label_weight if cfg.hard_label_weight is not None else (1.0 - cfg.alpha)
        return cfg.alpha * kd_loss + hard_weight * hard_loss


class DistillationTrainer(DistillationTrainerMixin, Trainer):
    """
    Trainer that performs knowledge distillation without adapter schedule updates.
    """

    def __init__(
        self,
        *args,
        teacher_model: Optional[nn.Module] = None,
        distillation_config: Optional[DistillationConfig] = None,
        **kwargs,
    ):
        self.distillation_config = distillation_config or DistillationConfig()
        super().__init__(*args, **kwargs)
        self._init_teacher(teacher_model, self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        hard_loss = outputs.loss
        total_loss = self.distillation_loss(student_logits, teacher_logits, labels, hard_loss)

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs

        return total_loss


class AdapterTrainerMixin:
    """
    Mixin that updates the AdapterSchedule for ExampleAdapter models.
    The consuming Trainer must expose `self.model` and call update_adapter_step().
    """

    adapter_update_interval: int = 1
    _adapter_step: int = 0

    def update_adapter_step(self, step: Optional[int] = None) -> Tuple[float, float]:
        if not isinstance(self.model, ExampleAdapterForCausalLM):
            raise TypeError("AdapterTrainerMixin requires the model to be ExampleAdapterForCausalLM.")

        if step is not None:
            self._adapter_step = step
            self.model.set_annealing_step(step)
        else:
            self._adapter_step += self.adapter_update_interval
            self.model.increment_annealing_step(self.adapter_update_interval)
        return self.model.get_current_weights()


class ExampleTrainer(DistillationTrainerMixin, AdapterTrainerMixin, Trainer):
    """
    Trainer that combines knowledge distillation and hybrid adapter annealing.
    """

    def __init__(
        self,
        *args,
        teacher_model: Optional[nn.Module] = None,
        distillation_config: Optional[DistillationConfig] = None,
        adapter_schedule: Optional[AdapterSchedule] = None,
        adapter_update_interval: int = 1,
        **kwargs,
    ):
        self.distillation_config = distillation_config or DistillationConfig()
        self.adapter_update_interval = adapter_update_interval

        super().__init__(*args, **kwargs)

        if isinstance(self.model, ExampleAdapterForCausalLM) and adapter_schedule is not None:
            self.model.adapter_schedule = adapter_schedule

        self._init_teacher(teacher_model, self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        hard_loss = outputs.loss
        total_loss = self.distillation_loss(student_logits, teacher_logits, labels, hard_loss)

        current_step = self.state.global_step if hasattr(self, "state") else None
        if isinstance(model, ExampleAdapterForCausalLM):
            if self.state.global_step > 0:
                self.update_adapter_step(current_step)

        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs

        return total_loss
