function [  ] = showFeatures( M, octave, circles )

for x=1:size(M,1)
        for y=1:size(M,2)
            if M(x,y)==1
                px = y*2^(octave-1);
                py = x*2^(octave-1);
                plot(px,py,'rx');
                if ((nargin > 2) && (circles == 1))
                    d = 5/2*2^octave;
                    h = rectangle('Position',[px-d/2 py-d/2 d d],'Curvature',[1,1],'EdgeColor','g');
                end
            end
        end
    end
end

