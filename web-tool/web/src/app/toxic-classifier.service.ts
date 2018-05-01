import { Injectable } from '@angular/core';

import axios from "axios";

@Injectable()
export class ToxicClassifierService {

  constructor() {

  }

  async getToxicity(text: string) {
    return axios.get("http://localhost:3000/toxic", {
      params: {
        text: text
      }
    });
  }

}
