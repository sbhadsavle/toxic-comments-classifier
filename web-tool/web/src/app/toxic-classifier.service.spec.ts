import { TestBed, inject } from '@angular/core/testing';

import { ToxicClassifierService } from './toxic-classifier.service';

describe('ToxicClassifierService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ToxicClassifierService]
    });
  });

  it('should be created', inject([ToxicClassifierService], (service: ToxicClassifierService) => {
    expect(service).toBeTruthy();
  }));
});
