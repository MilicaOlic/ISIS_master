import { Pipe, PipeTransform } from '@angular/core';

@Pipe({name: 'roundNumber'})
export class RoundNumberPipe implements PipeTransform {
    transform(input: any, args?: any): any {
        var exp, rounded,
          suffixes = ['k', 'M', 'G', 'T', 'P', 'E'];

        if (input.toString().includes('C'))
            return input;
        if (input.toString().includes('G'))
            return input;
        

        if (Number.isNaN(input)) {  
          return input;
        }
    
        if (input < 1000) {
          return input;
        }
    
        exp = Math.floor(Math.log(input) / Math.log(1000));
    
        return (input / Math.pow(1000, exp)).toFixed(args) + suffixes[exp - 1];
  }
}