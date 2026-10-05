const videos = [...document.querySelectorAll(".video-frame video")];

for (const video of videos) {
  // Keep native playback available if the player library cannot load.
  if (typeof Plyr !== "undefined") {
    new Plyr(video, {
      title: video.getAttribute("aria-label"),
      iconUrl: "vendor/plyr/plyr.svg",
      blankVideo: "vendor/plyr/blank.mp4",
      controls: [
        "play-large", "play", "progress", "current-time", "duration",
        "settings", "fullscreen",
      ],
      settings: ["speed"],
      speed: { selected: 1, options: [0.5, 1, 1.5, 2] },
      hideControls: false,
      invertTime: false,
      toggleTime: false,
      keyboard: { focused: true, global: false },
      tooltips: { controls: true, seek: true },
      storage: { enabled: false },
    });
  }

  video.addEventListener("play", () => {
    for (const other of videos) {
      if (other !== video) other.pause();
    }
  });
}
