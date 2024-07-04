import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class ClassificationService {

  private baseUrl = "http://localhost:8000/classify";

  constructor(private http: HttpClient) { }

  classifySentence(sentence: string, model: string): Observable<string> {
    return this.http.get<string>(`${this.baseUrl}/${sentence}?model=${model}`);
  }
}
