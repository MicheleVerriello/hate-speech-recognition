import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Feedback} from "../models/feedback";

@Injectable({
  providedIn: 'root'
})
export class ClassificationService {

  private baseUrl = "http://localhost:8000/classify";

  constructor(private http: HttpClient) { }

  classifySentence(sentence: string, model: string): Observable<string> {
    return this.http.get<string>(`${this.baseUrl}/${sentence}?model=${model}`);
  }

  sendFeedback(feedback: Feedback): Observable<any> {
    return this.http.post<any>(`${this.baseUrl}/feedback`, feedback);
  }
}
