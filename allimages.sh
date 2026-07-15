mkdir -p data/multipleview/VRroom1D/images

for cam in data/multipleview/VRroom1D/cam*; do
    cam_name=$(basename "$cam")  # e.g., cam01
    for img in "$cam"/*.png; do
        img_name=$(basename "$img")
        cp "$img" "data/multipleview/VRroom1D/images/${cam_name}_${img_name}"
    done
done
