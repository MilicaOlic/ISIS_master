import { Component } from '@angular/core';
import { ForecastService } from '../forecast.service';
import { ToastrService } from 'ngx-toastr';

@Component({
  selector: 'app-populate',
  templateUrl: './populate.component.html',
  styleUrls: ['./populate.component.css']
})
export class PopulateComponent {
  
  files:File[] = [];

  constructor(private service: ForecastService, private toastr: ToastrService) {}

  hasUpload(event:any){
    for (let index = 0; index < event.target.files.length; index++) {
      this.files.push(event.target.files[index.toString()]);
    }
  
  }

  uploadFiles() {
    const formData = new FormData();

    if(this.files.length == 0) {
      this.toastr.error('Select file first!');
      return;
    }

    formData.append('filesLength', this.files.length.toString());
    let index = 0;
    this.files.forEach(file => {
      formData.append('file[' + index.toString() + ']', <File>file, file.name);
      index++;
    });

    this.service.uploadFiles(formData).subscribe({
      next: (data) => {
        this.toastr.success('File upload successfull.');
      },
      error: (error) => {
        this.toastr.error('Error!');
      }
    });
  }
}
