/**
 * Layer for the borders of the Tetris board
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

public class Border extends Layer {
    /** Width of the board (should be >0) */
    int width;
    /** Height of the board */
    int height;

    /** Constructor of the border */
    public Border(int width, int height) {
        // Correct the parameter values
        if (width<=0) width = 1;
        if (height<=0) height = 1;
        // Set the parameter values
        this.width = width;
        this.height = height;
    }
    
    @Override
    public int get(int x, int y) {
        // Bias the walls so the accessible field are centered
        // or shifted to the right if width is even
        if (x==-(width-1)/2-1 || x==width/2+1 || y==0)
            return BLOCK;
        else
            return NONE;
    }
}