import { Component } from '@angular/core';
import { ToxicClassifierService } from './toxic-classifier.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  toxicityLabels = {
    toxic: "Toxic",
    severe_toxic: "Severe Toxic",
    obscene: "Obscene",
    threat: "Threat",
    insult: "Insult",
    identity_hate: "Identity Hate"
  };

  toxicityLabelKeys = Object.keys(this.toxicityLabels);

  text: string;
  title = 'app';
  toxicityResults = {
    "toxic": 0,
    "severe_toxic": 0,
    "obscene": 0,
    "threat": 0,
    "insult": 0,
    "identity_hate": 0
  };

  constructor(
    private toxicClassifierService: ToxicClassifierService
  ) {

  }

  async submitComment() {
    console.log(this.text);
    try {
      console.log("Sending request...");
      let response = await this.toxicClassifierService.getToxicity(this.text);
      console.log(response);
      this.toxicityResults = Object.assign(this.toxicityResults, response.data);
    }
    catch(err) {
      console.error(err);
    }

  }

}
