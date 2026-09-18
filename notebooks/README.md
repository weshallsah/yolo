# Training on Colab from VS Code

There is no supported way to attach VS Code to a Colab *kernel*. Colab's FAQ forbids
"remote control such as SSH shells, remote desktops", which is what `colab-ssh` and the
ngrok/cloudflared tunnels do, so this repo uses a git bridge instead: VS Code stays your
editor, git is the transport, Colab is only the GPU.

```
VS Code (edit, commit, push)  ->  github.com/weshallsah/yolo  ->  Colab GPU (git pull, train)
                                                                        |
                                            Google Drive  <-------------+  best.pt
```

## The loop

1. Edit `backend/scripts/...` in VS Code.
2. `git push`.
3. Open [`colab_train_retail.ipynb`](colab_train_retail.ipynb) in Colab, re-run cell 3 (it
   hard-resets to `origin/main`), then the training cell.
4. Weights land in Drive at `MyDrive/yolo-retail/retail_yolo_products_cls/`.

Cell 3 discarding Colab-side edits is deliberate — it guarantees you never train a version
that does not exist in git.

To open the notebook: push it, then visit
`https://colab.research.google.com/github/weshallsah/yolo/blob/main/notebooks/colab_train_retail.ipynb`.

## One-time setup

- **Kaggle token** — kaggle.com → Settings → API → Create New Token. Then either add
  `KAGGLE_USERNAME` and `KAGGLE_KEY` as **Colab Secrets** (key icon in the sidebar, with
  Notebook access enabled), or drop `kaggle.json` in Drive at `MyDrive/kaggle/kaggle.json`.
  Secrets are preferred: they outlive the runtime and keep the key out of Drive.
- **Accept the competition rules** at
  <https://www.kaggle.com/c/retail-products-classification/rules>. Competition downloads
  answer **401**, not 403, until you do — which reads like a rejected key and is the most
  common reason the download cell fails. Cell 6 probes a public endpoint first so it can tell
  you which of the two it actually is.

  This is an InClass competition. If the rules page offers no accept button, or the Data tab
  is unreachable in a browser, it is closed or limited to enrolled students and no token will
  open it. Use `--csv` and `--images-dir` to point the pipeline at another source.

## What gets trained

The competition has ~42k product images labelled into 21 categories, with **no bounding
boxes**. So the default is YOLOv8 *classification*:

```bash
python backend/scripts/train_on_retail.py --task classify
```

`--task detect` still works. It stamps one full-frame box (`0.5 0.5 0.98 0.98`) on every
image so the labels can train a detector. That shim is what makes the existing detection API
serve something, but the model only ever learns "one object fills the frame" and degrades
badly on real multi-product shelf photos. Prefer `classify` unless you specifically need the
old serving path.

## Serving the result

`backend/models/` is gitignored, so download `best.pt` from Drive into the repo, then:

```bash
# backend/.env
APP_YOLO_WEIGHTS_PATH=models/retail_yolo_products_cls/weights/best.pt
APP_MODEL_TASK=classify
```

`APP_MODEL_TASK=classify` is not optional. A `-cls` checkpoint produces `result.probs` and no
`result.boxes`, so the default detection path would return an empty list for every image
rather than failing loudly.

## Running on Kaggle instead

Colab's free GPU quota is unpredictable. The same scripts run unchanged in a Kaggle Notebook,
where the competition mounts at `/kaggle/input` with no token and no download step — add the
competition as a data source and run `train_on_retail.py`. `prepare_retail_dataset.py`
searches `/kaggle/input`, `/content/retail_data` and `backend/training_data/retail/raw`, so
it finds the data either way.
