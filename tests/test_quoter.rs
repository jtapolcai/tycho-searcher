mod tests {
    use std::collections::BTreeMap;
    use ordered_float::OrderedFloat;
    use tycho_searcher::searcher::graph_types::PriceData;

    use tycho_searcher::searcher::price_quoter::get_interpolated_y;

    /// Helper function to create a sorted Vec<(f64, PriceData)> from a slice of tuples.
    fn create_map(points_data: &[(f64, f64)]) -> BTreeMap<OrderedFloat<f64>, PriceData> {
        points_data.iter().map(|&(x, y)| (OrderedFloat(x), PriceData { amount_out: y, gas: 0.0 })).collect()
    }

    #[test]
    fn test_empty_map() {
        let points = create_map(&[]);
        assert_eq!(get_interpolated_y(&points, 5.0), None);
    }

    #[test]
    fn test_single_point_map() {
        let points = create_map(&[(10.0, 20.0)]);
        assert_eq!(get_interpolated_y(&points, 5.0), None);
        assert_eq!(get_interpolated_y(&points, 10.0), None); // Not enough neighbors for collinearity check
    }

    #[test]
    fn test_two_points_interpolation() {
        let points = create_map(&[(0.0, 0.0), (10.0, 10.0)]);
        assert_eq!(get_interpolated_y(&points, 5.0), Some(5.0));
        // Interpolate closer to one end
        assert_eq!(get_interpolated_y(&points, 2.5), Some(2.5));
        assert_eq!(get_interpolated_y(&points, 7.5), Some(7.5));

        let points_non_unit_slope = create_map(&[(0.0, 0.0), (10.0, 20.0)]);
        assert_eq!(get_interpolated_y(&points_non_unit_slope, 5.0), Some(10.0)); // y = 2x
        assert_eq!(get_interpolated_y(&points_non_unit_slope, 7.5), Some(15.0));
    }

    #[test]
    fn test_new_x_on_existing_point_collinear() {
        let points = create_map(&[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]);
        assert_eq!(get_interpolated_y(&points, 5.0), Some(5.0));

        let points_flat_line = create_map(&[(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)]);
        assert_eq!(get_interpolated_y(&points_flat_line, 5.0), Some(5.0));
    }

    #[test]
    fn test_new_x_on_existing_point_not_collinear() {
        let points = create_map(&[(0.0, 0.0), (5.0, 1.0), (10.0, 10.0)]);
        // Existing point at 5.0 is NOT collinear with its neighbors
        assert_eq!(get_interpolated_y(&points, 5.0), None);

        let points_convex_break = create_map(&[(0.0, 0.0), (5.0, 2.0), (10.0, 0.0)]);
        // A concave break, not collinear
        assert_eq!(get_interpolated_y(&points_convex_break, 5.0), None);
    }

    #[test]
    fn test_new_x_outside_range() {
        let points = create_map(&[(0.0, 0.0), (10.0, 10.0)]);
        assert_eq!(get_interpolated_y(&points, -5.0), None);
        assert_eq!(get_interpolated_y(&points, 15.0), None);
    }

    #[test]
    fn test_new_x_at_boundary_point() {
        let points = create_map(&[(0.0, 0.0), (10.0, 10.0)]);
        // At boundary, not enough distinct neighbors for collinearity check
        assert_eq!(get_interpolated_y(&points, 0.0), None);
        assert_eq!(get_interpolated_y(&points, 10.0), None);
    }
    #[test]
    fn test_three_points_linear_interpolation() {
        let points = create_map(&[(0.0, 0.0), (10.0, 20.0), (20.0, 40.0)]);
        assert_eq!(get_interpolated_y(&points, 5.0), Some(10.0));
        assert_eq!(get_interpolated_y(&points, 15.0), Some(30.0));
    }

    #[test]
    fn test_vertical_line_segment() {
        // This case is tricky for standard linear interpolation and collinearity.
        // For a function x->y, there should typically only be one y for an x.
        // If x1 and x2 are the same, it's not a segment.
        let points = create_map(&[(0.0, 0.0), (0.0, 10.0), (1.0, 1.0)]); // Invalid for a function normally
        assert_eq!(get_interpolated_y(&points, 5.0), Some(5.000000000000001)); // Within epsilon, should be considered collinear
    }
    #[test]
    fn test_slightly_off_collinear() {
        let points = create_map(&[(0.0, 0.0), (5.0, 5.000000000000001), (10.0, 10.0)]);
        assert_eq!(get_interpolated_y(&points, 5.0), Some(5.000000000000001)); // Within epsilon, should be considered collinear
    }
    #[test]
    fn test_no_lower_bound() {
        let points = create_map(&[(5.0, 5.0), (10.0, 10.0)]);
        assert_eq!(get_interpolated_y(&points, 2.0), None); // No lower bound for 2.0
    }

    #[test]
    fn test_no_upper_bound() {
        let points = create_map(&[(0.0, 0.0), (5.0, 5.0)]);
        assert_eq!(get_interpolated_y(&points, 7.0), None); // No upper bound for 7.0
    }
}
