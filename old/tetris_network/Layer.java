/**
 * Layer of objects/blocks
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

abstract public class Layer {
    /** Default block values
     * (negative values corresponds to positive blocks
     *  that have been frozen in place when falling) */
    public static final int BLOCK = -1;
    public static final int NONE = 0;
    public static final int TETRO = 1;
    public static final int I = 2;
    public static final int J = 3;
    public static final int L = 4;
    public static final int O = 5;
    public static final int S = 6;
    public static final int Z = 7;
    public static final int T = 8;

    /** Return the block value at the given coordinates
     *  @param x        x coordinate of the block to extract
     *  @param y        y coordinate of the block to extract
     *  @return         block value at these coordinates */
    abstract public int get(int x, int y);
    
    /** Test if there are violated constraints with the given layer
     * (i.e. are there overlaps between non empty blocks)
     * @param layer     layer to test against this one 
     * @param view      view to limit the search
     * @return          coordinates of the overlap or null if there is none */
    public int[] overlap(Layer layer, View view) {
        // Extract the views from both layers without modifying the original view
        View v1 = new View(view);
        v1.get(this);
        View v2 = new View(view);
        v2.get(layer);
        // Test if there are overlaps
        for (int x=0; x<view.width; x++) {
            for (int y=0; y<view.height; y++) {
                if (v1.blocks[x][y]!=NONE && v2.blocks[x][y]!=NONE)
                    return new int[]{x,y};
            }   
        }
        return null;
    }
}