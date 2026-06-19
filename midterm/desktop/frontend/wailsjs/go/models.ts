export namespace main {
	
	export class Answer {
	    answer: string;
	    prob: number;
	
	    static createFrom(source: any = {}) {
	        return new Answer(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.answer = source["answer"];
	        this.prob = source["prob"];
	    }
	}
	export class CheckpointsResp {
	    checkpoints: string[];
	    current: string;
	
	    static createFrom(source: any = {}) {
	        return new CheckpointsResp(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.checkpoints = source["checkpoints"];
	        this.current = source["current"];
	    }
	}
	export class HealthResp {
	    ready: boolean;
	    checkpoint: string;
	    has_attention: boolean;
	
	    static createFrom(source: any = {}) {
	        return new HealthResp(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.ready = source["ready"];
	        this.checkpoint = source["checkpoint"];
	        this.has_attention = source["has_attention"];
	    }
	}
	export class PredictResp {
	    answers: Answer[];
	    heatmap: string;
	    has_attention: boolean;
	
	    static createFrom(source: any = {}) {
	        return new PredictResp(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.answers = this.convertValues(source["answers"], Answer);
	        this.heatmap = source["heatmap"];
	        this.has_attention = source["has_attention"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}

}

