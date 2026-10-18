import json
import os

def save_settings(sv_dir, args):

    settings = vars(args)

    os.makedirs(f"results/{sv_dir}", exist_ok=True)

    with open(f"results/{sv_dir}/settings.json", "w") as file:
        json.dump(
            settings,
            file,
            sort_keys=True,
            indent=4
        )

