# Kaggle CLI workflow

The official Kaggle CLI is installed in this repository's existing Python virtual environment. Use the project wrapper so Kaggle always reads credentials from the private, git-ignored `.kaggle` directory rather than the Windows user profile.

## One-time authentication

1. Sign in to Kaggle.
2. Open **Settings** and find the API section.
3. Create/download the legacy API credential file named `kaggle.json`.
4. From the repository root, run:

   ```powershell
   .\scripts\kaggle-auth.ps1 -CredentialFile "C:\Users\Insha sadaf\Downloads\kaggle.json"
   ```

5. Verify authentication without displaying the secret:

   ```powershell
   .\scripts\kaggle.ps1 datasets list --mine
   ```

The `.kaggle` directory is excluded by `.gitignore`. Never paste the username/key JSON into a notebook, chat message, source file, or Git commit.

## Useful commands

Check the CLI:

```powershell
.\scripts\kaggle.ps1 --version
```

List your notebooks:

```powershell
.\scripts\kaggle.ps1 kernels list --mine
```

Check a notebook run:

```powershell
.\scripts\kaggle.ps1 kernels status inshasadaf114/NOTEBOOK-SLUG
```

Download all saved notebook outputs:

```powershell
.\scripts\kaggle.ps1 kernels output inshasadaf114/NOTEBOOK-SLUG -p artifacts/kaggle-output
```

Kernel upload metadata and push commands will be added after authentication confirms the exact Kaggle notebook slug and checkpoint-dataset slug.
