# Image Liquifier

Drag and drop your favorite images and watch as they are converted into a fluid/smoke which you can play around with!

## Build and Run
```bash
make
./bin/liquifier
```

## Controls
- **First, drag and drop any image of your choosing into the window.** Then:
- **Left Click**: Drag to apply velocity to the fluid.
- **Scroll Wheel**: Adjust brush radius.
- **Space**: Pause/Resume the movement of fluid.
- **R**: Reset the image.

---

## Technical Details
The  fluid and smoke simulation engine to create this is programmed entirely from scratch. The repo for just that part of the project is [here](https://github.com/grassEqualsBugs/SmokeSimulation), where you can learn more about the technical details.
