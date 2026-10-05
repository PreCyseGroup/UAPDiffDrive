const videos = [...document.querySelectorAll(".video-frame video")];

for (const button of document.querySelectorAll(".playback-toggle")) {
  const video = document.getElementById(button.getAttribute("aria-controls"));
  const updateButton = () => {
    const playing = !video.paused && !video.ended;
    button.textContent = playing ? "Pause video" : "Play video";
    button.setAttribute("aria-pressed", String(playing));
  };

  video.addEventListener("play", () => {
    for (const other of videos) {
      if (other !== video) other.pause();
    }
    updateButton();
  });
  video.addEventListener("pause", updateButton);
  video.addEventListener("ended", updateButton);
  button.addEventListener("click", async () => {
    if (!video.paused && !video.ended) {
      video.pause();
      return;
    }
    try {
      await video.play();
      button.removeAttribute("title");
    } catch {
      updateButton();
      button.title = "Unable to play. Try the video controls or download the MP4.";
    }
  });
  updateButton();
}
