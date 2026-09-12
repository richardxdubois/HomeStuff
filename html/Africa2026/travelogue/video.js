// Africa 2026 — shared video tile behavior.
//
// Each video tile in the markup looks like this:
//
//   <div class="tile video-tile" data-yt="YOUTUBE_VIDEO_ID" data-caption="Optional caption">
//     <img class="thumb" src="path/to/poster.jpg" alt="">
//     <span class="play-badge">...svg...</span>
//   </div>
//
// On scroll-into-view, the poster image + play badge are replaced with a
// YouTube iframe that autoplays muted (browsers block unmuted autoplay —
// this matches how Facebook/Instagram do feed autoplay: silent by default,
// the viewer can unmute via the player's own controls). Each tile only
// loads once; scrolling away does not tear it down again.

(function () {
  function buildEmbedSrc(videoId) {
    var params = [
      "autoplay=1",
      "mute=1",
      "playsinline=1",
      "controls=1",
      "rel=0",
      "modestbranding=1",
      "loop=1",
      "playlist=" + videoId // required by YouTube for single-video looping
    ].join("&");
    // youtube-nocookie.com is the privacy-enhanced embed domain, and
    // combined with an explicit referrerpolicy this avoids YouTube's
    // "error 153 / Watch on YouTube" banner, which shows up when the
    // embed can't see a proper referrer/origin (e.g. opening the page
    // as a bare file:// URL instead of through a server).
    return "https://www.youtube-nocookie.com/embed/" + videoId + "?" + params;
  }

  function activateTile(tile) {
    if (tile.dataset.loaded === "1") return;
    tile.dataset.loaded = "1";

    var videoId = tile.dataset.yt;
    if (!videoId) return;

    var iframe = document.createElement("iframe");
    iframe.src = buildEmbedSrc(videoId);
    iframe.setAttribute("allow", "autoplay; encrypted-media; picture-in-picture");
    iframe.setAttribute("allowfullscreen", "");
    iframe.setAttribute("referrerpolicy", "strict-origin-when-cross-origin");
    iframe.setAttribute("title", tile.dataset.caption || "video");

    tile.innerHTML = "";
    tile.appendChild(iframe);

    if (tile.dataset.caption) {
      var cap = document.createElement("div");
      cap.className = "video-caption";
      cap.textContent = tile.dataset.caption;
      tile.appendChild(cap);
    }
  }

  function init() {
    var tiles = document.querySelectorAll(".video-tile[data-yt]");
    if (!tiles.length) return;

    if (!("IntersectionObserver" in window)) {
      // fallback: just load them all
      tiles.forEach(activateTile);
      return;
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            activateTile(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );

    tiles.forEach(function (tile) {
      observer.observe(tile);
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
