"""Profile DA3METRIC-LARGE on one image."""

from profile_da3_common import profile_model


if __name__ == "__main__":
    profile_model("depth-anything/DA3METRIC-LARGE", "da3metric_large")

