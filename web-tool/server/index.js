
const Koa = require("koa");
const Router = require("koa-router");

const childProcess = require("child_process");

const stringify = require("csv-stringify");
const parse = require("csv-parse");

const fs = require("fs");

const collectFeaturesScript = "../../augmented_kernel/collect_feature_data.py";
const predictionScript = "../../augmented_kernel/persisted_kernel.py"

const columns = ["comment_text"]
const predictionColumns = [
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

function outputCsv(data) {
    return new Promise((resolve, reject) => {
        stringify(data, {
            header: true,
            columns: columns
        }, (err, output) => {
            if (err) {
                reject(err);
            }
            else {
                resolve(output);
            }
        });
    });
}

function parseCsv(data) {
    return new Promise((resolve, reject) => {
        parse(data, {
            columns: predictionColumns
        }, (err, output) => {
            if (err) {
                reject(err);
            }
            resolve(output);
        });
    })
}


function main() {
    let app = new Koa();
    let router = new Router();
    router.get("/toxic", async (ctx, next) => {
        console.log(ctx.query);
        // convert query to a single-element array
        let data = [[ctx.query.text]];
        let res = await outputCsv(data);
        fs.writeFileSync("input.csv", res);
        console.log(res);

        res = childProcess.spawnSync("python", [collectFeaturesScript, "-f", "input.csv", "-o", "input_augmented.csv"]);
        // console.log(res.stdout.toString());

        res = childProcess.spawnSync("python", [predictionScript, "-f", "input_augmented.csv", "-o", "prediction.csv"]);
        // console.log(res.stdout.toString());
        // console.log(res.stderr.toString());

        let prediction = fs.readFileSync("prediction.csv");

        let parsedPrediction = await parseCsv(prediction);
        console.log(parsedPrediction);

        // 0 is the header row, so we want prediction[1]
        prediction = parsedPrediction[1];

        ctx.body = prediction;

        ctx.response.code = 200;
    });

    app.use(router.routes());
    app.listen(3000);
}

main();
