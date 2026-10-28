# Deploy the shelf-recognition demo on a VPS

Step-by-step for a **public demo**: one FastAPI app serving shelf recognition, the SKU catalog, and the data dashboard.

This is not a production hardening guide (no auth). It is enough to put the app on a **CPU-only** Linux VPS and share a URL. There is no GPU on the server; the app runs on CPU (`faiss-cpu` + CPU PyTorch).

---

## What you are deploying

| URL | Page |
|-----|------|
| `http://YOUR_VPS/` | Shelf photo upload + detection + SKU match |
| `http://YOUR_VPS/catalog.html` | Manage SKU photos, visual search, add / delete images |
| `http://YOUR_VPS/dashboard.html` | Catalog KPIs and similarity |

The process loads **YOLO + DINOv3 + FAISS** once at startup, then serves every request from RAM.

---

## 1. Pick a VPS size (CPU only)

This demo is meant for a VPS **without a GPU**. Do not install NVIDIA drivers or CUDA.

| | Minimum | Comfortable |
|---|---|---|
| OS | Ubuntu 22.04 or 24.04 | same |
| RAM | **8 GB** | **16 GB** |
| Disk | 20 GB | 40 GB |
| GPU | **none** | **none** |
| vCPU | 2 | 4+ |

On CPU, a dense shelf photo can take **30–90 seconds**. Say that to demo viewers.

Do **not** train models on this VPS.

The pipeline is configured with `device = "cuda"` in code but **falls back to CPU** when CUDA is missing. That is the path you want on this server.

---

## 2. Runtime files (now tracked in git)

These used to be gitignored. They are required to start the app and are tracked:

| Path | Why |
|------|-----|
| `data/single_products/` | SKU gallery photos (`bottle/`, `can/`, `carton/`) |
| `gallery/faiss.index` | Vector index for SKU search |
| `gallery/metadata.json` | Index metadata (must match the embedder) |
| `runs/sku110k_s/weights/best.pt` | YOLO detector the web app loads |
| `runs/dinov3_ft/` | Fine-tuned DINOv3 embedder (`model.safetensors` is ~327 MB) |

Still **not** in git (not needed to run the demo): `data/shelf_images/`, `data/eval/`, `data/SKU-110K/`, other `runs/` training dumps, `tmp/`, `.venv/`.

**GitHub file-size note:** `runs/dinov3_ft/model.safetensors` is ~327 MB. GitHub rejects files over 100 MB unless you use [Git LFS](https://git-lfs.com):

```powershell
git lfs install
git lfs track "runs/dinov3_ft/model.safetensors"
git add .gitattributes runs/dinov3_ft/model.safetensors
```

If you skip LFS, clone the repo on the VPS and copy only that one file with `scp`.

---

## 3. On the VPS: create a user and install system packages

SSH in as root (or a sudo user), then:

```bash
# optional: dedicated user
adduser --disabled-password --gecos "" shelf
usermod -aG sudo shelf
su - shelf
```

Install Python, nginx, and OpenCV system libs:

```bash
sudo apt update
sudo apt install -y \
  python3.12 python3.12-venv python3-pip \
  git rsync unzip \
  nginx \
  libgl1 libglib2.0-0 \
  build-essential
```

If `python3.12` is missing on Ubuntu 22.04:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

---

## 4. Copy the project onto the VPS

Destination used below: `/home/shelf/ai-marchup-v2`. Prefer **git clone** now that the runtime files are tracked.

### Option A — git clone (preferred)

On the VPS, after you push the repo (with Git LFS if you tracked `model.safetensors`):

```bash
# Git LFS, so the 327 MB embedder actually downloads
sudo apt install -y git-lfs
git lfs install

git clone YOUR_REPO_URL /home/shelf/ai-marchup-v2
cd /home/shelf/ai-marchup-v2
mkdir -p tmp/web
```

If clone skipped LFS and `runs/dinov3_ft/model.safetensors` is a tiny pointer file, copy the real weights from your PC:

```powershell
scp runs\dinov3_ft\model.safetensors USER@YOUR_VPS_IP:/home/shelf/ai-marchup-v2/runs/dinov3_ft/
```

### Option B — rsync / scp from the PC

From WSL or Git Bash:

```bash
rsync -avh --progress \
  --exclude '.venv' --exclude '__pycache__' --exclude 'tmp' \
  --exclude 'data/SKU-110K' --exclude 'data/shelf_images' --exclude 'data/eval' \
  --exclude '.git' \
  ./ USER@YOUR_VPS_IP:/home/shelf/ai-marchup-v2/
```

### Option C — zip, upload, unzip

Zip the repo **without** `.venv`, `tmp`, `data/SKU-110K`. On the VPS:

```bash
cd /home/shelf
unzip ai-marchup-v2.zip
mkdir -p /home/shelf/ai-marchup-v2/tmp/web
```

---

## 5. Python environment and dependencies

```bash
cd /home/shelf/ai-marchup-v2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

This VPS has **no GPU**. Install the CPU wheel of PyTorch (not a CUDA build):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` already pins `faiss-cpu`. Do not install `faiss-gpu` or a CUDA PyTorch build here.

---

## 6. Hugging Face (only if the embedder is not fully local)

If `runs/dinov3_ft/` is a complete local checkpoint, skip this.

If startup tries to download `facebook/dinov3-vitb16-pretrain-lvd1689m`, that repo is **gated**. On the VPS:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the model license at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
3. Log in once:

```bash
source /home/shelf/ai-marchup-v2/.venv/bin/activate
pip install huggingface_hub
huggingface-cli login
```

Or put the token in the systemd unit (`Environment=HF_TOKEN=...`) — do not commit it.

---

## 7. Smoke-test before nginx

```bash
cd /home/shelf/ai-marchup-v2
source .venv/bin/activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Wait until you see something like:

```
Ready: detector=..., embedder=..., N gallery vectors, M catalog products
Application startup complete
```

First start can take **1–3 minutes** (model load). In another SSH session:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/
curl -s http://127.0.0.1:8000/api/catalog/skus | head
```

You want `200` and a JSON list of SKUs. Stop uvicorn with `Ctrl+C`.

If it crashes, see [Troubleshooting](#12-troubleshooting).

---

## 8. Run it as a systemd service

So the demo survives SSH disconnects and reboots.

```bash
sudo nano /etc/systemd/system/shelf.service
```

Paste:

```ini
[Unit]
Description=Shelf product recognition demo
After=network.target

[Service]
User=shelf
Group=shelf
WorkingDirectory=/home/shelf/ai-marchup-v2
Environment=PATH=/home/shelf/ai-marchup-v2/.venv/bin
# Optional: Environment=HF_TOKEN=hf_...
ExecStart=/home/shelf/ai-marchup-v2/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
# YOLO + DINOv3 need a long startup and a lot of RAM
TimeoutStartSec=180
MemoryMax=7G

[Install]
WantedBy=multi-user.target
```

**Use `--workers 1`.** Each worker loads its own copy of the models. Two workers on an 8 GB VPS will OOM.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now shelf
sudo systemctl status shelf
sudo journalctl -u shelf -f
```

Leave the log follow until `Application startup complete`, then `Ctrl+C`.

Useful later:

```bash
sudo systemctl restart shelf
sudo journalctl -u shelf -n 100 --no-pager
```

---

## 9. Put nginx in front (public HTTP)

Uvicorn stays on localhost. Nginx listens on port 80 (and later 443).

```bash
sudo nano /etc/nginx/sites-available/shelf
```

Paste (replace `YOUR_DOMAIN_OR_IP`):

```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    client_max_body_size 25M;   # matches the app upload limit

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # shelf analyze can be slow on CPU
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

Enable it:

```bash
sudo ln -sf /etc/nginx/sites-available/shelf /etc/nginx/sites-enabled/shelf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Open the firewall:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
sudo ufw status
```

On the cloud panel (Hetzner / DigitalOcean / AWS / OVH), also allow **inbound TCP 80** (and **443** if you add HTTPS). You do **not** need to expose port 8000.

Open `http://YOUR_VPS_IP/` in a browser.

---

## 10. HTTPS for a nicer demo (optional but recommended)

If you have a domain pointing at the VPS (`A` record → the server IP):

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d demo.yourdomain.com
```

Certbot edits the nginx site and auto-renews. After that, share `https://demo.yourdomain.com/`.

Without a domain, stay on `http://IP`. Browsers will show “not secure”; that is fine for an internal demo.

---

## 11. Demo checklist

1. **Shelf recognition** — `http://YOUR_VPS/`  
   Upload a shelf JPEG/PNG (max 25 MB). Wait for boxes + SKU cards.

2. **Catalog** — `http://YOUR_VPS/catalog.html`  
   Search SKUs, open one, add/delete photos. **Find by photo** uses the same embedder.

3. **Data dashboard** — `http://YOUR_VPS/dashboard.html`  
   KPIs only; it does not load GPU models by itself, but it is served by the same app.

Tell viewers:

- First click after a restart is slower (models already in RAM after systemd start; the *request* is still slow on CPU).
- Catalog writes go to `data/single_products/` on the server. They are **not** copied back to your PC unless you rsync them down.

---

## 12. Troubleshooting

| Symptom | What to do |
|---------|------------|
| `Missing gallery files` / FAISS error | `gallery/faiss.index` and `gallery/metadata.json` must be in the clone. They must have been built with `runs/dinov3_ft`. |
| `FileNotFoundError` for `best.pt` | `runs/sku110k_s/weights/best.pt` missing from the clone. That is `PipelineConfig.yolo_weights`. |
| Tiny `model.safetensors` (~100 bytes) | Git LFS pointer, not the real weights. Run `git lfs pull` or `scp` the 327 MB file. |
| Hugging Face gated / 401 | Log in with `huggingface-cli login`, or copy a complete `runs/dinov3_ft` folder so nothing is downloaded. |
| `Killed` / exit 137 / OOM | VPS RAM too small. Use 16 GB, keep `--workers 1`, close other services. |
| Upload 413 | nginx `client_max_body_size` is below 25M, or the file is larger than 25 MB. |
| Analyze times out | Raise `proxy_read_timeout` (already 300s above). CPU VPS + a huge photo is slow. |
| `libGL.so.1` / OpenCV import error | `sudo apt install -y libgl1 libglib2.0-0` |
| Blank page / 502 | `sudo systemctl status shelf` and `sudo journalctl -u shelf -n 80`. Nginx cannot reach uvicorn. |
| Catalog photos 404 | `data/single_products/{bottle\|can\|carton}/{sku}/` missing on the server. |
| Permission denied writing photos | `chown -R shelf:shelf /home/shelf/ai-marchup-v2` and ensure `tmp/web` exists. |

Quick health:

```bash
sudo systemctl is-active shelf
curl -I http://127.0.0.1:8000/
curl -I http://127.0.0.1/          # via nginx
```

---

## 13. Update the demo later

After you change code or gallery photos on the PC:

```bash
# from your PC (WSL / Git Bash)
rsync -avh --progress \
  --exclude '.venv' --exclude 'tmp' --exclude '__pycache__' \
  ./ USER@YOUR_VPS_IP:/home/shelf/ai-marchup-v2/
```

Then on the VPS:

```bash
sudo systemctl restart shelf
sudo journalctl -u shelf -f    # wait for Ready: ...
```

If you only added SKU photos on the **server** (Catalog page), the live FAISS index is updated in memory and written to `gallery/`. No restart needed for those.

If you copied a new `gallery/` from the PC, restart the service so it reloads the index.

---

## 14. Pull a copy of server-side catalog changes back to your PC

```powershell
# PowerShell on the PC
scp -r USER@YOUR_VPS_IP:/home/shelf/ai-marchup-v2/data/single_products ./data/
scp -r USER@YOUR_VPS_IP:/home/shelf/ai-marchup-v2/gallery ./
```

---

## Paths the running app expects

All relative to `/home/shelf/ai-marchup-v2`:

```
app.py
web/                          # HTML/CSS/JS
data/single_products/         # SKU images
gallery/faiss.index
gallery/metadata.json
runs/sku110k_s/weights/best.pt
runs/dinov3_ft/               # embedder
tmp/web/                      # created at runtime
```

That is the whole demo: clone the repo (plus Git LFS for the embedder), install **CPU** PyTorch, systemd, nginx, share the URL.
