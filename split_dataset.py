import os
import shutil
import random

DATASET = "dataset"

random.seed(42)

# Source folders
SOURCE = {
    "real": os.path.join(DATASET, "real"),
    "fake": os.path.join(DATASET, "fake")
}

# Destination folders
DESTINATIONS = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20
}


def clear_destination_folders():

    for split in DESTINATIONS:

        for label in ["real", "fake"]:

            folder = os.path.join(
                DATASET,
                split,
                label
            )

            os.makedirs(folder, exist_ok=True)

            # Remove old files
            for filename in os.listdir(folder):

                filepath = os.path.join(
                    folder,
                    filename
                )

                if os.path.isfile(filepath):
                    os.remove(filepath)


def get_video_id(filename):
    """
    Example:
    183_000.jpg -> 183
    253_001.jpg -> 253
    """

    name = os.path.splitext(filename)[0]

    return name.split("_")[0]


def split_class(label):

    source_folder = SOURCE[label]

    files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Group images by source video
    groups = {}

    for filename in files:

        video_id = get_video_id(filename)

        if video_id not in groups:
            groups[video_id] = []

        groups[video_id].append(filename)

    video_ids = list(groups.keys())

    random.shuffle(video_ids)

    total_videos = len(video_ids)

    train_count = max(
        1,
        int(total_videos * 0.60)
    )

    val_count = max(
        1,
        int(total_videos * 0.20)
    )

    train_ids = video_ids[:train_count]

    val_ids = video_ids[
        train_count:
        train_count + val_count
    ]

    test_ids = video_ids[
        train_count + val_count:
    ]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    print()
    print("===================================")
    print("Processing:", label.upper())
    print("===================================")

    for split_name, ids in splits.items():

        destination = os.path.join(
            DATASET,
            split_name,
            label
        )

        count = 0

        for video_id in ids:

            for filename in groups[video_id]:

                source = os.path.join(
                    source_folder,
                    filename
                )

                target = os.path.join(
                    destination,
                    filename
                )

                shutil.copy2(
                    source,
                    target
                )

                count += 1

        print(
            f"{split_name}: {count} images "
            f"from {len(ids)} videos"
        )


def main():

    print("===================================")
    print("DATASET SPLITTER")
    print("===================================")

    clear_destination_folders()

    split_class("real")
    split_class("fake")

    print()
    print("===================================")
    print("DATASET SPLIT COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()