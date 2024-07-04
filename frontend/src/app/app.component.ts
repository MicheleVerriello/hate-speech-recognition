import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {FormsModule} from "@angular/forms";
import {ClassificationService} from "./services/classification.service";
import {Feedback} from "./models/feedback";

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'Binary Classification';

  constructor(private classificationService: ClassificationService) {
  }

  sentence: string = '';
  model: string = '';
  result: string = '';

  classifySentence() {
    // removing characters that might create issues with tokenization
    this.sentence = this.cleanString(this.sentence);
    this.classificationService.classifySentence(this.sentence, this.model).subscribe(result => {
      this.result = result;
    });
  }

  cleanString(sentence: string): string {
    sentence = sentence.replaceAll('\"', "")
    sentence = sentence.replaceAll('\'', "")
    sentence = sentence.replaceAll('\\', "")
    sentence = sentence.replaceAll('\/', "")

    return sentence
  }

  sendFeedback(isResultCorrect: boolean) {
    const feedback: Feedback = {
      sentence: this.sentence,
      label: this.calculateResult(isResultCorrect)
    }
    this.classificationService.sendFeedback(feedback).subscribe(res => console.log(res));
  }

  calculateResult(isResultCorrect: boolean): string {
    if (isResultCorrect) return this.result;
    else if (this.result === "not sexist") { return "sexist"}
    else return "not sexist"

  }
}
