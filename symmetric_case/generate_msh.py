import os, subprocess, itertools, shutil
from numpy import deg2rad


def generate_mesh(args, template='geometry_2d.template_geo', dim=2):
    '''Modify template according args and make gmsh generate the mesh'''
    assert os.path.exists(template)
    args = args.copy()

    with open(template, 'r') as f: old = f.readlines()

    # Chop the file to replace the jet positions
    split = map(lambda s: s.startswith('DefineConstant'), old).index(True)

    jet_positions = deg2rad(map(float, args.pop('jet_positions')))
    jet_positions = 'jet_positions[] = {%s};\n' % (', '.join(map(str, jet_positions)))
    body = [jet_positions] + old[split:]

    # Make sure that args specifies all the constants (not necessary
    # as these have default values). This is more to check sanity of inputs
    last, _ = next(itertools.dropwhile(lambda (i, line): '];' not in line,
                                       enumerate(body)))
    constant_lines = body[1:last]
    constants = set(l.split('=')[0].strip() for l in constant_lines)
    body = "".join(body)  # Merge

    output = args.pop('output')

    if not output:
        output = template
    assert os.path.splitext(output)[1] == '.geo'

    with open(output, 'w') as f: f.write(body)

    args['jet_width'] = deg2rad(args['jet_width'])

    unrolled = '_'.join([output, 'unrolled'])
    shutil.copy(template, unrolled)
    assert os.path.exists(unrolled)

    scale = args.pop('clscale')

    list_geo_parameters = ['width', 'jet_radius', 'jet_width', 'length',
                           'bottom_distance', 'front_distance'];

    # Add parameters controlling mesh size
    list_geo_parameters.extend(['cylinder_bl_width', 'cylinder_inner_size', 'cylinder_outer_size',
                                'wake_length', 'wake_size',
                                'outlet_size', 'inlet_size'])
    # Original template
    try:
        assert set(list_geo_parameters) <= constants, (set(list_geo_parameters), constants)
    # Symmetric
    except AssertionError:
        list_geo_parameters.remove('jet_width')
        list_geo_parameters.remove('bottom_distance')

        assert set(list_geo_parameters) <= constants, (set(list_geo_parameters), constants)
        
    constants = ''
    # Commandline parameters for gmsh
    for p in list_geo_parameters:
        value = args.get(p, None)
        if value is not None:
            constants = constants + " -setnumber " + p + " " + str(value)

    return subprocess.call(['./gmsh -%d -clscale %g %s' % (dim, scale, unrolled)], shell=True)

# -------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys, petsc4py
    from math import pi

    parser = argparse.ArgumentParser(description='Generate msh file from GMSH',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optional output geo file
    parser.add_argument('-output', default='foo.geo', type=str, help='A geofile for writing out geometry')
    # Geometry
    parser.add_argument('-length', default=200, type=float,
                        help='Channel length')
    parser.add_argument('-front_distance', default=40, type=float,
                        help='Cylinder center distance to inlet')
    parser.add_argument('-bottom_distance', default=40, type=float,
                        help='Cylinder center distance from bottom wall')
    parser.add_argument('-jet_radius', default=10, type=float,
                        help='Cylinder radius')
    parser.add_argument('-width', default=80, type=float,
                        help='Channel width')
    # Jet perameters
    parser.add_argument('-jet_positions', nargs='+', default=[0, 60, 120, 180, 240, 300],
                        help='Angles of jet center points')
    parser.add_argument('-jet_width', default=10, type=float,
                        help='Jet width in degrees')

    # Refine geometry
    parser.add_argument('-clscale', default=1, type=float,
                        help='Scale the mesh size relative to give')

    args = parser.parse_args()

    # Using geometry_2d.geo to produce geometry_2d.msh
    sys.exit(generate_mesh(args.__dict__))
