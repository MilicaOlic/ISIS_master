import { Component } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ToastrService } from 'ngx-toastr';
import { TrainingDates } from 'src/app/types';
import { ForecastService } from '../forecast.service';

@Component({
  selector: 'app-train',
  templateUrl: './train.component.html',
  styleUrls: ['./train.component.css']
})
export class TrainComponent {

  constructor(private service: ForecastService, private toastr: ToastrService) {}

  dateMin = new Date(2018,1,1);
  dateMax = new Date(2021,9,1);

  dates = new FormGroup({
    dateFrom: new FormControl(this.dateMin, Validators.required),
    dateTo: new FormControl(this.dateMin, Validators.required),
  });

  trainModel() {
    if(this.dates.invalid) {
      this.toastr.error("Invalid dates entered!");
      return;
    }

    if (this.dates.controls.dateFrom.value! > this.dateMax || this.dates.controls.dateFrom.value! < this.dateMin) {
      this.toastr.error("Invalid dates entered!");
      return;
    }

    if (this.dates.controls.dateTo.value! > this.dateMax || this.dates.controls.dateTo.value! < this.dateMin) {
      this.toastr.error("Invalid dates entered!");
      return;
    }

    if (this.dates.controls.dateTo.value! < this.dates.controls.dateFrom.value!){
      this.toastr.error("Date from must be smaller than date to!");
      return;
    }

    let formData = new FormData();
    formData.append('dateFrom', this.dates.controls.dateFrom.value!.toString());
    formData.append('dateTo', this.dates.controls.dateTo.value!.toString());
    this.service.trainModel(formData).subscribe({
      next: (data) => {
        this.toastr.success('Training succesfull.');
      },
      error: (error) => {
        this.toastr.error('Error!');
      }
    });
  }

}
