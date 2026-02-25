#!/usr/bin/env bash

set -e

BASE_DIR="/data/why/spark_assets"

echo "Creating base directory: $BASE_DIR"
mkdir -p "$BASE_DIR"

download () {
  URL=$1
  DIR=$2
  FILE=$(basename "$URL")

  TARGET_DIR="$BASE_DIR/$DIR"
  mkdir -p "$TARGET_DIR"

  echo "⬇️  Downloading $FILE → $TARGET_DIR/"
  curl -L "$URL" -o "$TARGET_DIR/$FILE"
}

# ---------------- SPLATS ----------------
# download "https://sparkjs.dev/assets/splats/robot-head.spz" "splats"
# download "https://sparkjs.dev/assets/splats/forge.spz" "splats"
# download "https://sparkjs.dev/assets/splats/butterfly-wings-closed.spz" "splats"
# download "https://sparkjs.dev/assets/splats/butterfly.spz" "splats"
# download "https://sparkjs.dev/assets/splats/butterfly-ai.spz" "splats"
# download "https://sparkjs.dev/assets/splats/fireplace.spz" "splats"
# download "https://sparkjs.dev/assets/splats/snow-street.spz" "splats"
# download "https://sparkjs.dev/assets/splats/furry-logo-pedestal.spz" "splats"
# download "https://sparkjs.dev/assets/splats/cat.spz" "splats"
# download "https://sparkjs.dev/assets/splats/valley.spz" "splats"

# ------------ FOOD SPLATS ---------------
# download "https://sparkjs.dev/assets/splats/food/branzino-amarin.spz" "splats/food"
# download "https://sparkjs.dev/assets/splats/food/burger-from-amboy.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/clams-and-caviar-by-ikoyi.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/coral-caviar.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/double-Double-from-InNOut.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/gyro.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/iberico-sandwich-by-reserve.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/primerib-tamos.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/steaksandwich-mels.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/pad-thai.spz" "splats/food"
download "https://sparkjs.dev/assets/splats/food/tomahawk-niku.spz" "splats/food"

# ---------------- MODELS ----------------
download "https://sparkjs.dev/assets/models/rubberduck.glb" "models"
download "https://sparkjs.dev/assets/models/table.glb" "models"
download "https://sparkjs.dev/assets/models/arrow.glb" "models"

# ---------------- IMAGES ----------------
download "https://sparkjs.dev/assets/images/butterfly.png" "images"
download "https://sparkjs.dev/assets/images/sky.jpeg" "images"

echo "✅ All assets downloaded into ./$BASE_DIR"
