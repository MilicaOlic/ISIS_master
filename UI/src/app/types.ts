export interface CoalPowerPlant {
    id: number;
    name: string;
    power: number;
    co2: number;
    fuelPrice: number;
    priceToPowerCurve: number[][];
    priceToPowerCurveQuadratic: boolean;
    priceToCo2Curve: number[][];
    priceToCo2CurveQuadratic: boolean;
    Co2ToPowerCurve: number[][];
    Co2ToPowerCurveQuadratic: boolean;
}

export interface GasPowerPlant {
    id: number;
    name: string;
    power: number;
    co2: number;
    fuelPrice: number;
    priceToPowerCurve: number[][];
    priceToPowerCurveQuadratic: boolean;
    priceToCo2Curve: number[][];
    priceToCo2CurveQuadratic: boolean;
    Co2ToPowerCurve: number[][];
    Co2ToPowerCurveQuadratic: boolean;
}

export interface SolarPowerPlant {
    id: number;
    name: string;
    tiltAngle: number;
    area: number;
    efficiency: number;
}

export interface WindPowerPlant {
    id: number;
    name: string;
    numOfTurbines: number;
    bladeLength: number;
    maxPower: number;
}

export interface HydroPowerPlant {
    id: number;
    name: string;
    power: number;
    cost: number;
}

export interface OptimizationParameters {
    model: string;
    criteria: string;
    coalGenerators: CoalPowerPlant[];
    gasGenerators: GasPowerPlant[];
    solarGenerators: SolarPowerPlant[];
    windGenerators: WindPowerPlant[];
    hydroGenerators: HydroPowerPlant[];
}

export interface TrainingDates {
    dateFrom: string;
    dateTo: string;
}

export interface Response {
    list: Array<string>
    code: number
}