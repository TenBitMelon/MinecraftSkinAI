const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const http = require("http");
const puppeteer = require("puppeteer");

// const app = express();
const downloadFolder = path.join(__dirname, "downloadedskins");

// Create the download folder if it doesn't exist
if (!fs.existsSync(downloadFolder)) {
  fs.mkdirSync(downloadFolder);
}

// const server = http.createServer((req, res) => {
//   if (req.method === "POST") {
//     let data = "";
//     req.on("data", (chunk) => {
//       data += chunk;
//     });
//     req.on("end", async () => {
//       const imageUrl = JSON.parse(data).url;
//       res.end("Data received successfully.");

//       const response = await axios.get(imageUrl, { responseType: "stream" });
//       const fileName = path.basename(imageUrl);
//       const filePath = path.join(downloadFolder, fileName);
//       const writer = fs.createWriteStream(filePath);

//       response.data.pipe(writer);

//       writer.on("finish", () => {
//         console.log(`Image downloaded and saved to: ${filePath}`);
//       });

//       writer.on("error", (err) => {
//         console.error("Error while saving the image:", err);
//       });
//     });
//   } else {
//     res.statusCode = 404;
//     res.end();
//   }
// });

const collecskins = async () => {
  try {
    const browser = await puppeteer.launch({
      headless: false
    });
    try {
      const url = "https://www.namemc.com/minecraft-skins/random/";

      const page = await browser.newPage();
      await page.setViewport({ width: 1920, height: 1080 });
      await page.setUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15");
      await page.goto(url);
      await page.waitForSelector("div.card.mb-2>a");

      const skins = await page.evaluate(() => {
        return [...document.querySelectorAll("div.card.mb-2>a")].map((e) => e.href).map((e) => "https://s.namemc.com/i" + e.substring(e.lastIndexOf("/")) + ".png");
      });

      skins.forEach(async (skin) => {
        const imageUrl = skin;
        console.log(imageUrl);
        // res.end("Data received successfully.");

        const response = await axios.get(imageUrl, { responseType: "stream" });
        const fileName = path.basename(imageUrl);
        const filePath = path.join(downloadFolder, fileName);
        const writer = fs.createWriteStream(filePath);

        response.data.pipe(writer);

        writer.on("finish", () => {
          console.log(`Image downloaded and saved to: ${filePath}`);
        });

        writer.on("error", (err) => {
          console.error("Error while saving the image:", err);
        });
      });

      await new Promise((resolve) => setTimeout(resolve, 1000));
      // await page.close();
    } catch (e) {}
    await browser.close();
  } catch (e) {}
};

(async () => {
  while (true) {
    try {
      await Promise.all([collecskins(), collecskins(), collecskins(), collecskins(), collecskins()]);
    } catch (e) {}
  }
})();
