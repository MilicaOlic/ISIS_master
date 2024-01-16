import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ForecastComponent } from './forecast/forecast.component';
import { HomeComponent } from './home/home.component';

const routes: Routes = [
  {path:'', component:ForecastComponent},
  {path:'forecast', component:ForecastComponent},
  {
    path:"**",
    redirectTo:''
  }
  
];

@NgModule({
  imports: [RouterModule.forRoot(routes, {
    onSameUrlNavigation: 'reload'
  })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
