/**
 * Compound layer
 * (to ease the combination of several layers of objects)
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

import java.util.*;

public class CompoundLayer extends Layer {
    /** Set of sub layers */
    LinkedList<Layer> layers = new LinkedList<Layer>();

    @Override
    public int get(int x, int y) {
        // Iterator in reverse order as to consider first the front layers
        // (the background corresponds to the first layer)
        Iterator<Layer> iter = layers.descendingIterator();
        while (iter.hasNext()) {
            int block = iter.next().get(x,y);
            // Only return the block without considering the others,
            // if this blocks actually contains something for this layer
            if (block != NONE)
                return block;
        }
        // No block has been found
        return NONE;
    }
    
    /** Add a layer to this compound layer
     * (utility method) */
    public void add(Layer layer) {
        layers.add(layer);
    }
}