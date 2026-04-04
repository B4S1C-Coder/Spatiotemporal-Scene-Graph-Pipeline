# Environment Setup Script

Use `infra/setup_env.sh` from the repository root to bootstrap the local environment.

It performs four setup steps:

- creates the expected repository directories
- creates `venv/` if it does not exist
- installs `requirements.txt` into the virtual environment
- creates `.env` from `.env-example` or built-in defaults if needed

Run it with:

```bash
bash infra/setup_env.sh
```

Useful overrides:

```bash
DRY_RUN=1 bash infra/setup_env.sh
SKIP_PIP_INSTALL=1 bash infra/setup_env.sh
UPGRADE_PIP=0 bash infra/setup_env.sh
```

The script creates or uses these paths by default:

- `venv/`
- `.env`
- `data/visdrone/`
- `weights/`
- `logs/`
- `notebooks/`
