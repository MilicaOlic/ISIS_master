import { DatePipe } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ToastrService } from 'ngx-toastr';
import { ForecastService } from './forecast.service';

@Component({
  selector: 'app-forecast',
  templateUrl: './forecast.component.html',
  styleUrls: ['./forecast.component.css']
})
export class ForecastComponent implements OnInit {

  constructor(private service: ForecastService, private toastr: ToastrService, public datepipe: DatePipe) { }

  models:any;
  predicted = false;
  data = [];
  dates = [];
  chartOptions = {
    title: {
    text: "Load"             
    },
    animationEnabled: true,
    axisX: {      
      //valueFormatString: ""
    },
    toolTip: {
    shared: true,
    contentFormatter: function (e: any) {
      var content = '';
      for (var i = e.entries.length - 1; i >= 0; i--) {
      content += "<span style ='color:" + e.entries[i].dataSeries.color + "; font-weight: bold;';>" + e.entries[i].dataSeries.name + "</span>:" + e.entries[i].dataPoint.y + "MW";
      content += "<br/>";
      }
      return content;
    }
    },
    legend: {
    cursor: "pointer",
    itemclick: function(e: any) {
      if (typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
      e.dataSeries.visible = false;
      }
      else {
      e.dataSeries.visible = true;
      }
      e.chart.render();
    }
    },
    data: [
      {
        type: "stackedArea",
        name: "Prediction",
        showInLegend: true,
        legendMarkerType: "square",
        markerSize: 0,
        color: "rgba(0,0,255,.9)",
        dataPoints: [
          {x: new Date(), y: 1}
          ]
      }            
    ]
  }
  showGraph = false;

  ngOnInit(): void {
    this.service.getModels().subscribe({
      next: (data) => {
        this.models = data as Array<string>;
        this.models = this.models.data;
      },
      error: (error) => {
        this.toastr.error('Error! Backend not started!');
      }
    });

    this.chartOptions = {
      title: {
      text: "Predicted electricity load"             
      },
      animationEnabled: true,
      axisX: {      
        valueFormatString: "DD-DDD-HH"
      },
      toolTip: {
      shared: true,
      contentFormatter: function (e: any) {
        var content = '';
        for (var i = e.entries.length - 1; i >= 0; i--) {
          content += e.entries[i].dataPoint.x.toLocaleString();
          content += "<br/>";
          content += "<span style ='color:" + e.entries[i].dataSeries.color + "; font-weight: bold;';>" + e.entries[i].dataSeries.name + "</span>:" + e.entries[i].dataPoint.y + "MW";
          content += "<br/>";
        }
        return content;
      }
      },
      legend: {
      cursor: "pointer",
      itemclick: function(e: any) {
        if (typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
        e.dataSeries.visible = false;
        }
        else {
        e.dataSeries.visible = true;
        }
        e.chart.render();
      }
      },
      data: [
        {
          type: "column",
          markerSize: 5,
          name: "Prediction",
          showInLegend: true,
          legendMarkerType: "triangle",
          color: "rgba(82, 170, 71, 0.5)",
          dataPoints: [
            {x: new Date(), y: 1}
            ]
        }            
      ]
    }

    this.chartOptions['data'][0]['dataPoints'] = []
    let time;
    if(this.showGraph) {
      this.data.forEach((element: number, index: number) => {
        time = new Date(this.dates[index]);
        time.setHours(time.getHours() - 2);
        (this.chartOptions['data'][0]['dataPoints']).push({x: new Date(time), y: element})
      });
    }

  }

  dateMin = new Date(2018,1,1);

  testForm = new FormGroup({
    model: new FormControl(0, Validators.required),
    date: new FormControl(this.dateMin, Validators.required),
    days: new FormControl(0, Validators.required),
  });

  test() {
    if (this.testForm.invalid) {
      this.toastr.error('Fill all required fields!');
      return;
    }

    if (this.testForm.controls.model.value! == 0) {
      this.toastr.error('Fill all required fields!');
      return;
    }

    if (this.testForm.controls.days.value! == 0) {
      this.toastr.error('Fill all required fields!');
      return;
    }

    // npm install ng2-charts chart.js --save

    let formData = new FormData();
    formData.append('model', this.testForm.controls.model.value!.toString());
    formData.append('date', this.testForm.controls.date.value!.toString());
    formData.append('days', this.testForm.controls.days.value!.toString());

    this.service.test(formData).subscribe({
      next: (data) => {
        this.toastr.success('Test finished!');
        this.data = data.data;
        this.dates = data.dates;
        this.predicted = true;

        this.showGraph = true;
        this.ngOnInit();
      },
      error: (error) => {
        this.toastr.error('Error!');
        this.predicted = false;
      }
    });
  }

  csv() {
    this.service.csv().subscribe({
      next: (data) => {
        this.toastr.success('CSV saved!');
      },
      error: (error) => {
        this.toastr.error(error.error.error);
      }
    });
  }

}
