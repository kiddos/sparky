from PIL import Image, ImageSequence
import argparse


def create_sprite_sheet_from_webp(webp_path, output_path, target_size=(256, 256), loop_back=False):
    """
    Extracts and resizes frames from a WebP animation, creates a sprite sheet,
    and optionally adds a loop back of frames in reverse order.

    Args:
        webp_path (str): The path to the input WebP animation file.
        output_path (str): The path where the output sprite sheet will be saved.
        target_size (tuple): The target size (width, height) for each frame.
        loop_back (bool): If True, adds the frames in reverse order to the sprite sheet.
    """
    try:
        # Open the WebP file
        with Image.open(webp_path) as img:
            # Check if it's an animated WebP
            if not getattr(img, "is_animated", False):
                print("Error: The provided file is not an animated WebP.")
                return

            # Get and resize all the frames
            resized_frames = [frame.copy().resize(target_size) for frame in ImageSequence.Iterator(img)]

            frames_to_combine = resized_frames

            # Add loop back if requested
            if loop_back:
                frames_to_combine = resized_frames + resized_frames[-2::-1]  # Exclude the last frame to avoid repetition

            # Determine the size of the final sprite sheet
            width = sum(frame.width for frame in frames_to_combine)
            height = frames_to_combine[0].height

            # Create a new blank image for the sprite sheet
            sprite_sheet = Image.new('RGBA', (width, height))

            # Paste each frame into the new image
            x_offset = 0
            for frame in frames_to_combine:
                sprite_sheet.paste(frame, (x_offset, 0))
                x_offset += frame.width

            # Save the sprite sheet
            sprite_sheet.save(output_path, 'PNG')
            print(f"Successfully created sprite sheet at {output_path} with size {target_size} and loop back: {loop_back}")

    except FileNotFoundError:
        print(f"Error: The file '{webp_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a sprite sheet from an animated WebP file with resizing and optional loop back.')
    parser.add_argument('input', type=str, help='The path to the input WebP animation file.')
    parser.add_argument('output', type=str, help='The path where the output sprite sheet will be saved.')
    parser.add_argument('--size', type=int, nargs=2, default=[128, 128], help='The target size (width height) for each frame (e.g., --size 128 128).')
    parser.add_argument('--loop', action='store_true', help='Add a loop back of the frames in reverse order to the sprite sheet.')
    args = parser.parse_args()

    create_sprite_sheet_from_webp(args.input, args.output, tuple(args.size), args.loop)
