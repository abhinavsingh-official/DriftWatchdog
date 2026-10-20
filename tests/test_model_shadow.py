from model_shadow import load_model, load_reference_and_holdout, score_model


def test_load_reference_and_holdout_splits_are_disjoint_and_sized_correctly():
    (X_ref, y_ref), (X_hold, y_hold), feature_cols = load_reference_and_holdout(test_size=0.3)
    assert X_ref.shape[0] + X_hold.shape[0] == 1440
    assert X_hold.shape[0] == round(1440 * 0.3)
    assert len(feature_cols) == X_ref.shape[1] == 180


def test_real_model_scores_undrifted_holdout_highly():
    model = load_model()
    (_, _), (X_hold, y_hold), feature_cols = load_reference_and_holdout()
    acc = score_model(model, X_hold, y_hold, feature_cols)
    assert acc > 0.9
