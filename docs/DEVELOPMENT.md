# Development guide

Two machines, no GitHub required (yet).

## Quick start (Linux, this machine)

```bash
cd ~/Documents/gpubasedpostrgress/duckdbgpumetaldb
./scripts/build.sh
./build-linux/test/test_gpudb
./build-linux/bin/gpudb-bench --rows 50000000 --runs 5
```

If `nvcc` is missing the build is CPU-only — that's fine for now. Install CUDA toolkit when ready (see `scripts/install_cuda_ubuntu.sh`).

## Quick start (macOS)

```bash
cd ~/dev/duckdbgpumetaldb     # wherever you place it
./scripts/build.sh
./build-macos/test/test_gpudb
./build-macos/bin/gpudb-bench --rows 50000000 --runs 5
```

Metal backend is auto-enabled on macOS. Xcode Command Line Tools are required (`xcode-select --install`).

## Moving the repo to your Mac without GitHub

Pick whichever fits — both keep the repo private to you and the two machines.

### Option A: rsync over SSH
On the Mac, after enabling Remote Login in System Settings:
```bash
# from Linux:
rsync -avz --exclude build-linux --exclude data --exclude .tools \
    ~/Documents/gpubasedpostrgress/duckdbgpumetaldb/ \
    you@your-mac.local:~/dev/duckdbgpumetaldb/
```

### Option B: tarball + AirDrop / USB
```bash
# from Linux:
cd ~/Documents/gpubasedpostrgress
tar --exclude='duckdbgpumetaldb/build-linux' \
    --exclude='duckdbgpumetaldb/data' \
    --exclude='duckdbgpumetaldb/.tools' \
    -czf duckdbgpumetaldb.tgz duckdbgpumetaldb/
# AirDrop or scp duckdbgpumetaldb.tgz to the Mac, then:
#   tar -xzf duckdbgpumetaldb.tgz && cd duckdbgpumetaldb && ./scripts/build.sh
```

Both options preserve the local git history. On the Mac side:
```bash
cd duckdbgpumetaldb
git status              # should be clean
git log --oneline       # should match Linux history
```

## Two-machine git workflow without a remote

If you want both machines to share commits but skip GitHub for now, you can use either:

### Direct SSH push between machines
```bash
# from Mac, pull from Linux box (one-time):
git remote add linux ssh://you@linux-box/home/you/Documents/gpubasedpostrgress/duckdbgpumetaldb
git pull linux main

# subsequent pulls:
git pull linux main
```

This requires SSH access between the boxes but never touches GitHub.

### USB / AirDrop bundle
```bash
# Linux side (sender):
git bundle create /tmp/sync.bundle main
# move sync.bundle to Mac via AirDrop/USB
# Mac side (receiver):
git fetch /Volumes/USB/sync.bundle main:linux-main
git merge linux-main      # or rebase, your call
```

## When you're ready to push to GitHub

The repo `singhpratech/duckdbgpumetaldbram` already exists and is private. To start using it:

```bash
cd ~/Documents/gpubasedpostrgress/duckdbgpumetaldb
git remote add origin https://github.com/singhpratech/duckdbgpumetaldbram.git
git push -u origin main
```

That's it. Until you run that command nothing leaves your machines.

## Branch / commit conventions
See [CLAUDE.md](../CLAUDE.md). Short version:
- Never commit to `main` directly (well — first commit excepted). Use `feat/cuda-*`, `feat/metal-*`, `feat/core-*`, `chore/*`.
- Conventional commit messages: `feat(cuda): add radix sort kernel`, `fix(cpu): handle empty input`.
- Test before committing: `./scripts/build.sh && ./build-*/test/test_gpudb`.
