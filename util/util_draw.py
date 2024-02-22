"""Help functions for showing results via plots"""

import os
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import gridspec


def draw_history(epoch: int, tr: np.array, val: np.array, isacc: bool):
    """Draw training history for loss or accuracy

    Args:
        epoch (int): Training epoch
        tr: Value series for training
        val: Value series for validating
        isacc (bool): True if draw accuracy history, False if show loss
    """
    name = 'Loss'
    if isacc:
        name = 'Acc'
    epochs = [i + 1 for i in range(epoch)]
    plt.plot(epochs, tr, 'r+-', label='Training')
    plt.plot(epochs, val, 'b.-', label='Validation')
    plt.title('Training and validation ' + name)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.legend()
    plt.show()


def draw_roc(y_score: np.array, y_label: np.array,
             threshold: float, pos_label: int = 1,
             print_prf: bool = True):
    """Draw ROC (Receiver Operating Characteristic) curve.

    Args:
        y_score: Output confidence scores given by the model
        y_label: Ground truth labels
        threshold (float): Threshold to differentiate similar and dissimilar
        pos_label (int): Label for positive (similar) samples, default is 1
        print_prf (bool): Print precision, recall and F1 score if required
    """
    from sklearn import metrics
    fpr_ui, tpr_ui, _ = metrics.roc_curve(y_label, y_score, pos_label=pos_label)
    auc_ui = metrics.auc(fpr_ui, tpr_ui)
    plt.figure()
    plt.plot(fpr_ui, tpr_ui, 'g-', label=f'AUC={auc_ui:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title(f'ROC curve')
    plt.legend(loc="lower right")
    if print_prf:
        y_pred = [1 if i > threshold else 0 for i in y_score] if pos_label == 1 else \
            [0 if i > threshold else 1 for i in y_score]
        print(f'p: {metrics.precision_score(y_label, y_pred, pos_label=pos_label)}')
        print(f'r: {metrics.recall_score(y_label, y_pred, pos_label=pos_label)}')
        print(f'f1: {metrics.f1_score(y_label, y_pred, pos_label=pos_label)}')
    plt.show()


def get_saliency_map(x1: torch.Tensor, x2: torch.Tensor,
                     y: torch.Tensor,
                     device, net: nn.Module):
    """Get saliency map"""
    net.eval()
    x1, x2 = x1.float().to(device), \
             x2.float().to(device)
    x1.requires_grad_()
    x2.requires_grad_()
    y.requires_grad_()

    # forward
    x = torch.stack((x1, x2), 0)
    o = net(x)
    r = int(o.shape[0] / 2)
    o1, o2 = o[:r, :], o[r:, :]
    score = torch.cosine_similarity(o1, o2)
    score = torch.clamp(score, min=1e-10)
    if score.item() <= 0.5:
        score = score * 2
    else:
        score = (score - 0.5) * 2
    score.backward()

    grads1, grads2 = x1.grad.abs(), x2.grad.abs()

    grads1, grads2 = grads1.cpu().numpy(), grads2.cpu().numpy()
    grads1, grads2 = np.squeeze(grads1), np.squeeze(grads2)

    return grads1, grads2


def draw_saliency_map(ylist, plist, hash1s, hash2s):
    """Draw saliency map

    Args:
        ylist: Ground truths
        plist: Output scores
        hash1s: Input UI# (column 1)
        hash2s: Input UI# (column 2)
    """
    _n = len(ylist)

    def _rerange(x, minmax: (float, float)):
        if np.max(x) == np.min(x):
            return np.ones(shape=np.shape(x)) * minmax[0]
        s = (minmax[1] - minmax[0]) / (np.max(x) - np.min(x))
        return (x - np.min(x)) * s + minmax[0]

    def _getimg(a):
        # horizontal concatenate
        rows = []
        for j in range(2):
            row = [a[k + 5 * j] for k in range(5)]
            row_n = []
            for k in range(5):
                # in the used cmap bar, the larger a value, the lighter the color
                kk = _rerange(-row[k], (k * .2 + 0.08, (k + 1) * .2 - 0.08))
                row_n.append(kk)
            row_nn = np.concatenate(row_n, axis=1)
            rows.append(row_nn)
        # vertial concatenate
        return np.concatenate(rows)

    for i in range(_n):
        # salincy
        plt.subplot(8, _n, i + 1)
        map_ui1 = _getimg(plist[i][0])
        plt.imshow(map_ui1[0:3, :], cmap=plt.cm.get_cmap('tab20b'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n + 1)
        plt.imshow(map_ui1[3:6, :], cmap=plt.cm.get_cmap('tab20c'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n * 2 + 1)
        map_ui2 = _getimg(plist[i][1])
        plt.imshow(map_ui2[0:3, :], cmap=plt.cm.get_cmap('tab20b'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n * 3 + 1)
        plt.imshow(map_ui2[3:6, :], cmap=plt.cm.get_cmap('tab20c'))
        plt.axis('off')

        # ui hash
        plt.subplot(8, _n, i + _n * 4 + 1)
        hash1, hash2 = _getimg(hash1s[i]), _getimg(hash2s[i])
        plt.imshow(hash1[0:3, :], cmap=plt.cm.get_cmap('tab20b'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n * 5 + 1)
        plt.imshow(hash1[3:6, :], cmap=plt.cm.get_cmap('tab20c'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n * 6 + 1)
        plt.imshow(hash2[0:3, :], cmap=plt.cm.get_cmap('tab20b'))
        plt.axis('off')

        plt.subplot(8, _n, i + _n * 7 + 1)
        plt.imshow(hash2[3:6, :], cmap=plt.cm.get_cmap('tab20c'))
        plt.axis('off')

        plt.title(ylist[i])

    plt.show()


def interactive(
        points,
        labels=None,
        values=None,
        hover_data=None,
        theme=None,
        cmap="Blues",
        color_key=None,
        color_key_cmap="Spectral",
        background="white",
        width=800,
        height=800,
        point_size=None,
        subset_points=None,
        interactive_text_search=False,
        interactive_text_search_columns=None,
        interactive_text_search_alpha_contrast=0.95,
        len_black=None
):
    """Show HCA (Hierarchical Clustering Analysing) results
    via an interactive plot"""
    from bokeh.layouts import column
    from bokeh.models import CustomJS
    from bokeh.models.widgets.inputs import TextInput
    from umap.plot import _themes, _to_hex
    if theme is not None:
        cmap = _themes[theme]["cmap"]
        color_key_cmap = _themes[theme]["color_key_cmap"]
        background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    if point_size is None:
        point_size = 100.0 / np.sqrt(points.shape[0])

    data = pd.DataFrame(points, columns=("x", "y"))

    if labels is not None:
        data["label"] = labels

        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )

        if isinstance(color_key, dict):
            data["color"] = pd.Series(labels).map(color_key)
        else:
            unique_labels = np.unique(labels)
            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {k: color_key[i] for i, k in enumerate(unique_labels)}
            data["color"] = pd.Series(labels).map(new_color_key)

        if len_black is not None:
            data["color"][-len_black:] = "#000000"
        colors = "color"

    elif values is not None:
        data["value"] = values
        palette = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        import bokeh.transform as btr
        colors = btr.linear_cmap(
            "value", palette, low=np.min(values), high=np.max(values)
        )

    else:
        colors = matplotlib.colors.rgb2hex(plt.get_cmap(cmap)(0.5))

    if subset_points is not None:
        data = data[subset_points]
        if hover_data is not None:
            hover_data = hover_data[subset_points]

    if points.shape[0] <= width * height // 10:

        if hover_data is not None:
            tooltip_dict = {}
            for col_name in hover_data:
                data[col_name] = hover_data[col_name]
                tooltip_dict[col_name] = "@{" + col_name + "}"
            tooltips = list(tooltip_dict.items())
        else:
            tooltips = None

        data["alpha"] = 1

        import bokeh.plotting as bpl
        data_source = bpl.ColumnDataSource(data)

        plot = bpl.figure(
            width=width,
            height=height,
            tooltips=tooltips,
            background_fill_color=background,
        )

        plot.axis.major_label_text_font = 'Cambria'
        plot.axis.major_label_text_font_size = '36px'
        plot.grid.visible = True
        plot.axis.visible = True

        plot.circle(
            x="x",
            y="y",
            source=data_source,
            color=colors,
            size=point_size,
            alpha="alpha",
        )

        if interactive_text_search:
            text_input = TextInput(value="", title="Search:")

            if interactive_text_search_columns is None:
                interactive_text_search_columns = []
                if hover_data is not None:
                    interactive_text_search_columns.extend(hover_data.columns)
                if labels is not None:
                    interactive_text_search_columns.append("label")

            if len(interactive_text_search_columns) == 0:
                print(
                    "interactive_text_search_columns set to True, but no hover_data or labels provided."
                    "Please provide hover_data or labels to use interactive text search."
                )

            else:
                callback = CustomJS(
                    args=dict(
                        source=data_source,
                        matching_alpha=interactive_text_search_alpha_contrast,
                        non_matching_alpha=1 - interactive_text_search_alpha_contrast,
                        search_columns=interactive_text_search_columns,
                    ),
                    code="""
                    var data = source.data;
                    var text_search = cb_obj.value;

                    var search_columns_dict = {}
                    for (var col in search_columns){
                        search_columns_dict[col] = search_columns[col]
                    }

                    // Loop over columns and values
                    // If there is no match for any column for a given row, change the alpha value
                    var string_match = false;
                    for (var i = 0; i < data.x.length; i++) {
                        string_match = false
                        for (var j in search_columns_dict) {
                            if (String(data[search_columns_dict[j]][i]).includes(text_search) ) {
                                string_match = true
                            }
                        }
                        if (string_match){
                            data['alpha'][i] = matching_alpha
                        }else{
                            data['alpha'][i] = non_matching_alpha
                        }
                    }
                    source.change.emit();
                """,
                )

                text_input.js_on_change("value", callback)

                plot = column(text_input, plot)
    else:
        if hover_data is not None:
            print(
                "Too many points for hover data -- tooltips will not"
                "be displayed. Sorry; try subssampling your data."
            )
        if interactive_text_search:
            print(
                "Too many points for text search." "Sorry; try subssampling your data."
            )
        import holoviews as hv
        import holoviews.operation.datashader as hd
        import datashader as ds
        hv.extension("bokeh")
        hv.output(size=300)
        hv.opts('RGB [bgcolor="{}", xaxis=None, yaxis=None]'.format(background))
        if labels is not None:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("color"),
                color_key=color_key,
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        elif values is not None:
            min_val = data.values.min()
            val_range = data.values.max() - min_val
            data["val_cat"] = pd.Categorical(
                (data.values - min_val) // (val_range // 256)
            )
            point_plot = hv.Points(data, kdims=["x", "y"], vdims=["val_cat"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count_cat("val_cat"),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )
        else:
            point_plot = hv.Points(data, kdims=["x", "y"])
            plot = hd.datashade(
                point_plot,
                aggregator=ds.count(),
                cmap=plt.get_cmap(cmap),
                width=width,
                height=height,
            )

    return plot
