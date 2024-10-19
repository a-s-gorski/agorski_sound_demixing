import argparse
from pipelines.inference.separate import separate_dir, separate_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_separate = subparsers.add_parser("separate_file")
    parser_separate.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="The config file of a model being trained.",
    )
    parser_separate.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path."
    )
    parser_separate.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="The path of audio to be separated.",
    )
    parser_separate.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to write out separated audio.",
    )
    parser_separate.add_argument(
        '--scale_volume',
        action='store_true',
        default=False,
        help="Set this flag to scale separated audios to maximum value of 1.",
    )
    parser_separate.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help="Set this flag to use CPU.",
    )

    parser_separate_dir = subparsers.add_parser("separate_dir")
    parser_separate_dir.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="The config file of a model being trained.",
    )
    parser_separate_dir.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path."
    )
    parser_separate_dir.add_argument(
        "--audios_dir",
        type=str,
        required=True,
        help="The directory of audios to be separated.",
    )
    parser_separate_dir.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="The directory to write out separated audios.",
    )
    parser_separate_dir.add_argument(
        '--scale_volume',
        action='store_true',
        default=False,
        help="Set this flag to scale separated audios to maximum value of 1.",
    )
    parser_separate_dir.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help="Set this flag to use CPU.",
    )

    args = parser.parse_args()

    if args.mode == "separate_file":
        separate_file(args)

    elif args.mode == "separate_dir":
        separate_dir(args)

    else:
        raise NotImplementedError
