document.getElementById("save-btn").addEventListener("click", () => {
    const canvas = document.getElementById("drawingCanvas");
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append("image", blob, "drawing.png");

        fetch("/save_drawing", {
            method: "POST",
            body: formData
        })
        .then(response => response.text())
        .then(data => alert(data))
        .catch(err => console.error(err));
    });
});
