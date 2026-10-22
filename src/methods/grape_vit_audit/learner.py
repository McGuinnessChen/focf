"""Use the preserved GrAPE learning rule; add forward-only diagnostics."""
from methods.grape.learner import MacroGrAPELearner


class AuditedGrAPELearner(MacroGrAPELearner):
    def step(self, images, labels):
        logits, loss, diagnostics = super().step(images, labels)
        diagnostics.update(self.model.activation_diagnostics)
        return logits, loss, diagnostics
