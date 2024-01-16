import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ForecastComponent } from './forecast/forecast.component';
import { HomeComponent } from './home/home.component';
import * as echarts from 'echarts';
import { NgxEchartsModule } from 'ngx-echarts';
import { NgxSliderModule } from '@angular-slider/ngx-slider';
import { PopulateComponent } from './forecast/populate/populate.component';
import { HttpClientModule } from '@angular/common/http';
import { ToastrModule } from 'ngx-toastr';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { MatExpansionModule } from '@angular/material/expansion';
import { TrainComponent } from './forecast/train/train.component';
import { DatePipe } from '@angular/common';

import * as CanvasJSAngularChart from '../assets/canvasjs.angular.component';
import { RoundNumberPipe } from './round-number.pipe';
var CanvasJSChart = CanvasJSAngularChart.CanvasJSChart;

@NgModule({
  declarations: [
    AppComponent,
    ForecastComponent,
    HomeComponent,
    PopulateComponent,
    TrainComponent,
    CanvasJSChart,
    RoundNumberPipe
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    NgxEchartsModule.forRoot({
      echarts
    }),
    NgxSliderModule,
    HttpClientModule,
    BrowserAnimationsModule,
    ToastrModule.forRoot(),
    ReactiveFormsModule,
    FormsModule,
    MatExpansionModule
  ],
  providers: [DatePipe],
  bootstrap: [AppComponent]
})
export class AppModule { }
