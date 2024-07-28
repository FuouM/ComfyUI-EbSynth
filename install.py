import os
import shutil

# https://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/install.py

subpack_path = os.path.join(os.path.dirname(__file__), "Ezsynth")
subpack_repo = "https://github.com/FuouM/Ezsynth.git"


def ensure_subpack():
    import git

    if os.path.exists(subpack_path):
        try:
            repo = git.Repo(subpack_path)
            repo.remotes.origin.pull()
        except Exception as e:
            print(e)
            import platform
            import traceback

            traceback.print_exc()
            if platform.system() == "Windows":
                print(
                    f"[ComfyUI-EbSynth] Please turn off ComfyUI and remove '{subpack_path}' and restart ComfyUI."
                )
            else:
                shutil.rmtree(subpack_path)
                git.Repo.clone_from(subpack_repo, subpack_path)
    else:
        git.Repo.clone_from(subpack_repo, subpack_path)

if __name__ == "__main__":
    ensure_subpack()
