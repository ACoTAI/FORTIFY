from fortify_scg.eval import precision_recall_f1_acc

def test_metrics_shape():
    y_true = [0,1,1,0,1]
    y_hat  = [0,1,0,0,1]
    m = precision_recall_f1_acc(y_true, y_hat, num_classes=2)
    assert set(m.keys()) == {"precision_macro","recall_macro","f1_macro","accuracy"}
