
    package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.Isomap;
    /*
        n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    
    */

    import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

    public class OptionsFor_Isomap_n_jobs extends NumericRangeOptionPredicate {
        
        @Override
        protected double getMin() {
            return 2;
        }

        @Override
        protected double getMax() {
            return 4;
        }

        @Override
        protected int getSteps() {
            return 1;
        }

        @Override
        protected boolean needsIntegers() {
            return true;
        }
    }
    
