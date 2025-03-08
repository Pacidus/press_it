import argparse
import os
import shutil
import subprocess
import tempfile

libjpeg_turbo = "/opt/libjpeg-turbo/bin/cjpeg"
mozjpeg = "/opt/mozjpeg/bin/cjpeg"
magick = "magick"
extention = {"mozjpeg": "jpg", "webp": "webp", "avif": "avif"}


def check_tools(tools):
    missing = []
    for tool in tools:
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        raise RuntimeError(f"Missing required tools: {', '.join(missing)}")


def get_ssim2(original_path, compressed_path):
    try:
        result = subprocess.run(
            ["as2c", original_path, compressed_path],
            capture_output=True,
            text=True,
        )
        ssim = float(result.stdout)
        print(f"\n\tssim={ssim}\n\n")
        return ssim
    except (subprocess.CalledProcessError, ValueError):
        return None


def get_ssim(original_path, compressed_path):
    try:
        result = subprocess.run(
            ["compare", "-metric", "ssim", original_path, compressed_path, "null:"],
            capture_output=True,
            text=True,
        )
        ssim = float(result.stderr.split("(")[1].split(")")[0])
        print(f"\n\tssim={ssim}\n\n")
        return ssim
    except (subprocess.CalledProcessError, ValueError):
        return None


get_ssims = [get_ssim, get_ssim2]


def find_best(program):
    ext = extention[program.__name__[1:]]

    def binary_search(original, tmp_dir, target_ssim):
        low, high, best = 0, 100, (None, None)
        for _ in range(7):
            if low > high:
                break
            mid = int(0.5 + ((low + high) / 2))
            output = os.path.join(tmp_dir, f"{program.__name__}_{mid}.{ext}")
            try:
                decoded_png = program(original, output, mid)
                ssim = ssims(original, decoded_png)
                if ssim and ssim >= target_ssim:
                    best = (mid, os.path.getsize(output))
                    high = mid
                else:
                    low = mid
            except subprocess.CalledProcessError:
                continue
        return best

    return binary_search


@find_best
def cwebp(input_png, output_webp, quality):
    subprocess.run(
        ["cwebp", "-m", "6", "-q", str(quality), input_png, "-o", output_webp],
        check=True,
    )
    decoded_png = os.path.join(os.path.dirname(output_webp), "decoded_webp.png")
    subprocess.run(["dwebp", output_webp, "-o", decoded_png], check=True)
    return decoded_png


@find_best
def cavif(input_png, output_avif, quality):
    subprocess.run(
        [
            "avifenc",
            "-q",
            f"{quality}",
            "-j",
            "all",
            "-s",
            "0",
            input_png,
            output_avif,
        ],
        check=True,
    )
    decoded_png = os.path.join(os.path.dirname(output_avif), "decoded_avif.png")
    subprocess.run(["avifdec", output_avif, decoded_png], check=True)
    return decoded_png


@find_best
def cmozjpeg(input_png, output_jpg, quality):
    subprocess.run(
        [mozjpeg, "-quality", str(quality), "-outfile", output_jpg, input_png],
        check=True,
    )
    decoded_png = os.path.join(os.path.dirname(output_jpg), "decoded_mozjpeg.png")
    subprocess.run([magick, output_jpg, decoded_png], check=True)
    return decoded_png


def process_png(reference_png, temp_dir):
    output_png = os.path.join(temp_dir, "compressed.png")
    subprocess.run(["pngcrush", reference_png, output_png], check=True)
    return os.path.getsize(output_png), 100


def main():
    global ssims
    parser = argparse.ArgumentParser(description="Optimize image for target SSIM.")
    parser.add_argument("input_image", help="Input image path")
    parser.add_argument("target_ssim", type=float, help="Target SSIM (0.0-1.0)")
    parser.add_argument("--ssim2", type=int, default=0, help="ssim 2 ??")
    args = parser.parse_args()
    ssims = get_ssims[args.ssim2]
    try:
        check_tools(
            [
                magick,
                "cwebp",
                "dwebp",
                "avifenc",
                "avifdec",
                "cjpeg",
                "pngcrush",
                "compare",
            ]
        )
    except RuntimeError as e:
        print(e)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        reference_png = os.path.join(temp_dir, "reference.png")
        subprocess.run(["magick", args.input_image, reference_png], check=True)

        results = {}

        # MozJPEG
        quality, size = cmozjpeg(reference_png, temp_dir, args.target_ssim)
        if quality:
            results["mozjpeg"] = {"size": size, "quality": quality}
        print(results["mozjpeg"])

        # WebP
        quality, size = cwebp(reference_png, temp_dir, args.target_ssim)
        print(quality, size)
        if quality:
            results["webp"] = {"size": size, "quality": quality}

        # AVIF
        quality, size = cavif(reference_png, temp_dir, args.target_ssim)
        if quality:
            results["avif"] = {"size": size, "quality": quality}

        # PNG
        try:
            size, quality = process_png(reference_png, temp_dir)
            results["png"] = {"size": size, "quality": quality}
        except:
            pass

        if not results:
            print("No valid conversions found.")
            return
        formats = {i: results[i]["size"] for i in results}
        best_format = min(formats, key=formats.get)
        out = f"""
            Best format: {best_format}
            ({results[best_format]['size']} bytes)
        """
        print(out)
        print(results)
        ext = extention[best_format]
        qual = results[best_format]["quality"]
        best = os.path.join(temp_dir, f"c{best_format}_{qual}.{ext}")
        name = os.path.basename(args.input_image).split(".")[0]
        shutil.move(best, f"./{name}_{qual}_{args.target_ssim}.{ext}")

    return results


if __name__ == "__main__":
    results = main()
