
import numpy as np

from base import TransformerMixin


class SelectorMixin(TransformerMixin):
    def transform(self, X, threshold=None):
            importances = self.feature_importances_
            if importances is None:
                raise ValueError("Importance weights not computed. Please  "
                    "Set the compute_importances parameter before fit.")

        elif hasattr(self, "coef_"):
            if self.coef_.ndim == 1:
                importances = np.abs(self.coef_)

            else:
                importances = np.sum(np.abs(self.coef_), axis=0)

        else:
            raise ValueError("Missing `feature_importances_` or `coef_`"
                             " attribute, did you forget to set the "
                             "estimator's parameter to compute it?")

        # Retrieve threshold
        if threshold is None:
            threshold = getattr(self, "threshold", "mean")

        if isinstance(threshold, basestring):
            if "*" in threshold:
                scale, reference = threshold.split("*")
                scale = float(scale.strip())
                reference = reference.strip()

                if reference == "median":
                    reference = np.median(importances)
                elif reference == "mean":
                    reference = np.mean(importances)
                else:
                    raise ValueError("Unknown reference: " + reference)

                threshold = scale * reference

            elif threshold == "median":
                threshold = np.median(importances)

            elif threshold == "mean":
                threshold = np.mean(importances)

        else:
            threshold = float(threshold)

        # Selection
        mask = importances >= threshold

        if np.any(mask):
            return X[:, mask]

        else:
            raise ValueError("Invalid threshold: all features are discarded.")
