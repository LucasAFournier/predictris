/**
 * Predictor model
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

import java.util.*;

public class Predictor {
    /** Predictions */
    int[][] predictions;
    
    /** Number of inputs */
    int input_dim;

    /** Constant for the content */
    public static final int EMPTY = 0;
    public static final int FULL = 1;
    public static final int UNDEF = 2;

    /** Constructor for the predictor model */
    public Predictor(int input_dim, int motor_nb) {
        this.input_dim = input_dim;
        // Create a big enough array so all possible input/motor configurations
        // can be represented (high space taken, but efficient access)
        predictions = new int[(int)Math.pow(UNDEF+1,input_dim)*motor_nb][];
    }

    /** Linearize and filter the sensory signals only */
    public int[] filter(int[][] in) {
        return filter(in,null);
    }

    /** Linearize and filter the sensory signals only */
    public int[] filter(int[][] in, int[] out) {
        // Create a long enough 1D array
        if (out==null)
            out = new int[in.length*in[0].length];
        // Fill the array and filter the values from the sensory signal
        for (int x=0; x<in.length; x++) {
            for (int y=0; y<in[0].length; y++) {
                // Here do not make any difference between the type of elements
                // (so the model only differentiates filled and empty blocks)
                out[x*in[0].length+y] = in[x][y]==Layer.NONE ? EMPTY : FULL;
            }
        }
        return out;
    }

    /** Update the predictions */
    public void update(int[][] cont, int mot, int[][] cons) {
        update(filter(cont),mot,filter(cons));        
    }
    
    /** Update the predictions (with context vector and consequences) */
    public void update(int[] cont, int mot, int[] cons) {
        // Must consider all combination of sensory input
        // for each signal, it can take its actual value or be undefined
        
        // Start from the original input array
        int[] c = Arrays.copyOf(cont,cont.length);
        // Start from the end
        int i=c.length-1;
        while(i>=0) {
            // If we are at the last sensory signal
            if (i==c.length-1) {
//                 for(int j=0; j<c.length; j++)
//                     System.out.print(c[j] + " ");
//                 System.out.println();

                // then generate the index to access past predictions
                int ind = indexOf(c,mot);
                // and test the dynamics against past knowledge/experience
                // TODO
                predictions[ind] = merge(cons,predictions[ind]);
            }
            // Test if we already changed this value
            if (c[i]==UNDEF) { // then revert this one and go back
                c[i]=cont[i];
                i--;
            } else { // then change it start again from the end
                c[i]=UNDEF;
                i=c.length-1;
            }
        }
    }
    
    /** Convert the input and motor command into an index */
    public int indexOf(int[] cont, int mot) {
        int ind = 0;
        for (int i=0; i<cont.length; i++) {
            ind = ind*(UNDEF+1)+cont[i];
        }
        // Motor command as the main entry
        // (to help get contiguous elements as it is mandatory)
        ind *= mot;
        return ind;
    }
    
    /** Convert the input and motor command into an index */
    public int[] arrayOf(int ind) {
        int[] cont = new int[input_dim+1]; // with motor signal
        // Retrieve the sensory signals
        for (int i=0; i<input_dim; i++) {
            cont[input_dim-1-i] = ind%(UNDEF+1);
            ind /= UNDEF+1;
        }
        // Get the motor signal
        cont[input_dim] = ind;
        return cont;
    }
    
    /** Integrate the observed dynamics within past predictions */
    public int[] merge(int[] obs, int[] past) {
        // Test if no predictions were ever made about this dynamics
        if (past==null) {
            return obs;
        } else { // merge the predictions
            for (int i=0; i<obs.length; i++) {
                if (obs[i]!=past[i])
                    past[i]=UNDEF;
            }
            return past;
        }
    }
    
    /** Display predictions */
    public void display() {
        System.out.println("Predictions");
        for (int p=0; p<predictions.length; p++) {
            // Check if the prediction has been defined
            if (predictions[p]!=null) {
                // Check if the prediction predicts something
                boolean pred = false;
                for (int i=0; i<predictions[p].length && !pred; i++)
                    pred |= predictions[p][i]!=UNDEF;
                if (pred) {
                    // Display the prediction
                    System.out.println(
                        Arrays.toString(arrayOf(p)) + "->"
                      + Arrays.toString(predictions[p])
                    );
                }
            }
        }
    }
}